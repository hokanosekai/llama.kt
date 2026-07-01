package com.tensai.llamakt.example

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import com.tensai.llamakt.BackendInfo
import com.tensai.llamakt.ChatMessage
import com.tensai.llamakt.LlamaEngine
import com.tensai.llamakt.chat
import kotlinx.coroutines.*
import java.io.File

private val PRESET_PROMPTS = listOf(
    "Hello, who are you?",
    "Explain the HyperLogLog algorithm in detail.",
    "Write a Python script to compute the Fibonacci sequence, with an explanation.",
    "Summarize the theory of general relativity for a beginner.",
    "Write a detailed essay about the history of computing.",
    "List 20 creative startup ideas, each with a one-line description.",
)

// ---------------------------------------------------------------------------
// System metric helpers
// ---------------------------------------------------------------------------

/** Read VmRSS from /proc/self/status — resident set size in MB. IO thread only. */
private fun readRamMb(): Float? = try {
    File("/proc/self/status").bufferedReader().useLines { lines ->
        lines.firstOrNull { it.startsWith("VmRSS:") }
            ?.removePrefix("VmRSS:")
            ?.trim()
            ?.split("\\s+".toRegex())
            ?.firstOrNull()
            ?.toLongOrNull()
            ?.let { kb -> kb / 1024f }
    }
} catch (_: Exception) { null }

/**
 * CPU % for this process, computed as:
 *   delta_ticks / (CLK_TCK * delta_wall_seconds) * 100
 *
 * May exceed 100% on multi-threaded workloads — by design, not capped.
 * CLK_TCK is typically 100 on Android (USER_HZ = 100).
 *
 * Returns null on first call (no previous snapshot) or on error.
 */
private object CpuSampler {
    private val clkTck: Long = try {
        android.system.Os.sysconf(android.system.OsConstants._SC_CLK_TCK)
    } catch (_: Exception) { 100L }

    private var prevTicks: Long = -1L
    private var prevWallMs: Long = -1L

    fun sample(): Float? {
        return try {
            val stat = File("/proc/self/stat").readText().trim()
            // The second field (comm) can contain spaces wrapped in parens; find the closing ')' first.
            val closeParenIdx = stat.lastIndexOf(')')
            val fields = stat.substring(closeParenIdx + 2).trim().split(' ')
            // After ')': state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt utime stime ...
            // fields[0]=state, fields[11]=utime, fields[12]=stime
            val utime = fields[11].toLong()
            val stime = fields[12].toLong()
            val totalTicks = utime + stime
            val nowMs = System.currentTimeMillis()

            val prev = prevTicks
            val prevMs = prevWallMs
            prevTicks = totalTicks
            prevWallMs = nowMs

            if (prev < 0L || prevMs < 0L) return null  // first call — no delta yet

            val deltaTicks = totalTicks - prev
            val deltaMs = nowMs - prevMs
            if (deltaMs <= 0L) return null

            val deltaWallSec = deltaMs / 1000.0
            (deltaTicks.toDouble() / (clkTck * deltaWallSec) * 100.0).toFloat()
        } catch (_: Exception) { null }
    }

    fun reset() {
        prevTicks = -1L
        prevWallMs = -1L
    }
}

/**
 * Best-effort GPU % via Mali sysfs.
 * Returns null permanently once determined unavailable (avoids repeated IO).
 *
 * Paths tried (in order):
 *   /sys/class/misc/mali0/device/utilization
 *   /sys/devices/platform/[platform].mali/utilization  (first match via File.listFiles glob)
 *   /sys/kernel/gpu/gpu_busy
 *   /sys/class/kgsl/kgsl-3d0/gpubusy         (Adreno fallback)
 */
private object GpuSampler {
    // null = untried, true/false = determined
    private var available: Boolean? = null
    private var resolvedPath: String? = null

    fun isUnavailable(): Boolean = available == false

    private fun findPath(): String? {
        val candidates = listOf(
            "/sys/class/misc/mali0/device/utilization",
            "/sys/kernel/gpu/gpu_busy",
            "/sys/class/kgsl/kgsl-3d0/gpubusy",
        )
        for (path in candidates) {
            val f = File(path)
            if (f.exists() && f.canRead()) return path
        }
        // Platform glob: /sys/devices/platform/*.mali/utilization
        try {
            val platformDir = File("/sys/devices/platform")
            platformDir.listFiles { f -> f.isDirectory && f.name.endsWith(".mali") }
                ?.firstOrNull()
                ?.let { mali ->
                    val util = File(mali, "utilization")
                    if (util.exists() && util.canRead()) return util.absolutePath
                }
        } catch (_: Exception) {}
        return null
    }

    fun sample(): Float? {
        if (available == false) return null
        if (available == null) {
            resolvedPath = findPath()
            available = resolvedPath != null
            if (available == false) return null
        }
        return try {
            val raw = File(resolvedPath!!).readText().trim()
            // Format may be "42" or "42 %" or "42/100" — extract first number
            raw.split("\\s+|/".toRegex()).firstOrNull()?.toFloatOrNull()
        } catch (_: Exception) {
            available = false
            null
        }
    }
}

class MainActivity : ComponentActivity() {

    private val engine = LlamaEngine()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { BenchScreen() }
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.free()
    }

    /**
     * Generic live metric graph, reused for tok/s, RAM, CPU, GPU.
     *
     * @param samples      list of sampled float values (mutableStateListOf)
     * @param label        e.g. "RAM (MB)"
     * @param unit         e.g. "MB" — shown in the placeholder text
     * @param primaryColor line / grid color
     * @param unavailable  if true, show a "N/A" banner instead of the graph
     */
    @Composable
    fun LiveGraph(
        samples: List<Float>,
        label: String,
        unit: String,
        primaryColor: Color,
        modifier: Modifier = Modifier,
        unavailable: Boolean = false,
    ) {
        val textMeasurer = rememberTextMeasurer()

        Column(modifier = modifier) {
            // Label + current / avg row
            val current = samples.lastOrNull()
            val avg = if (samples.isNotEmpty()) samples.average().toFloat() else null
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(label, style = MaterialTheme.typography.labelSmall, color = primaryColor)
                if (unavailable) {
                    Text(
                        "N/A — needs root",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outlineVariant,
                    )
                } else if (current != null && avg != null) {
                    Text(
                        "now %.1f  avg %.1f %s".format(current, avg, unit),
                        style = MaterialTheme.typography.labelSmall,
                        color = primaryColor.copy(alpha = 0.75f),
                    )
                }
            }

            // Graph area
            val graphModifier = Modifier
                .fillMaxWidth()
                .weight(1f)

            if (unavailable) {
                Box(modifier = graphModifier, contentAlignment = Alignment.Center) {
                    Text(
                        "GPU: N/A — permission denied (no root)",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outlineVariant,
                    )
                }
                return@Column
            }

            if (samples.isEmpty()) {
                Box(modifier = graphModifier, contentAlignment = Alignment.Center) {
                    Text(
                        "(graph will appear during generation)",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outlineVariant,
                    )
                }
                return@Column
            }

            Canvas(modifier = graphModifier) {
                val w = size.width
                val h = size.height
                val padLeft = 48f
                val padRight = 8f
                val padTop = 4f
                val padBottom = 4f

                val graphW = w - padLeft - padRight
                val graphH = h - padTop - padBottom

                val maxVal = (samples.max() * 1.15f).coerceAtLeast(1f)

                val gridColor = primaryColor.copy(alpha = 0.12f)
                val labelColor = primaryColor.copy(alpha = 0.55f)
                val gridSteps = 3
                for (i in 0..gridSteps) {
                    val fy = padTop + graphH * (1f - i.toFloat() / gridSteps)
                    drawLine(gridColor, Offset(padLeft, fy), Offset(padLeft + graphW, fy), strokeWidth = 1f)
                    val labelVal = maxVal * i / gridSteps
                    val labelText = if (labelVal >= 10f) "%.0f".format(labelVal) else "%.1f".format(labelVal)
                    val measured = textMeasurer.measure(
                        labelText,
                        style = TextStyle(fontSize = 9.sp, color = labelColor)
                    )
                    drawText(measured, topLeft = Offset(padLeft - measured.size.width - 4f, fy - measured.size.height / 2f))
                }

                // Polyline
                if (samples.size >= 2) {
                    val path = Path()
                    samples.forEachIndexed { idx, v ->
                        val x = padLeft + graphW * idx / (samples.size - 1).toFloat()
                        val y = padTop + graphH * (1f - (v / maxVal).coerceIn(0f, 1f))
                        if (idx == 0) path.moveTo(x, y) else path.lineTo(x, y)
                    }
                    drawPath(path, color = primaryColor, style = Stroke(width = 2.5f))

                    val lastX = padLeft + graphW
                    val lastY = padTop + graphH * (1f - (samples.last() / maxVal).coerceIn(0f, 1f))
                    drawCircle(primaryColor, radius = 4f, center = Offset(lastX, lastY))
                } else {
                    val x = padLeft + graphW / 2f
                    val y = padTop + graphH * (1f - (samples.first() / maxVal).coerceIn(0f, 1f))
                    drawCircle(primaryColor, radius = 4f, center = Offset(x, y))
                }

                // Average dashed line
                val avgVal = samples.average().toFloat()
                val avgY = padTop + graphH * (1f - (avgVal / maxVal).coerceIn(0f, 1f))
                val dashLen = 6f
                val gapLen = 4f
                var x = padLeft
                while (x < padLeft + graphW) {
                    val x2 = (x + dashLen).coerceAtMost(padLeft + graphW)
                    drawLine(primaryColor.copy(alpha = 0.35f), Offset(x, avgY), Offset(x2, avgY), strokeWidth = 1f)
                    x += dashLen + gapLen
                }
                val avgText = "avg %.1f".format(avgVal)
                val avgMeasured = textMeasurer.measure(
                    avgText,
                    style = TextStyle(fontSize = 9.sp, color = primaryColor.copy(alpha = 0.55f))
                )
                drawText(avgMeasured, topLeft = Offset(padLeft + graphW - avgMeasured.size.width - 2f, avgY - avgMeasured.size.height - 2f))
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun BenchScreen() {
        // State
        var localPath by remember { mutableStateOf<String?>(null) }
        var modelLoaded by remember { mutableStateOf(false) }
        var prompt by remember { mutableStateOf("Hello, who are you?") }
        var output by remember { mutableStateOf("") }
        var status by remember { mutableStateOf("Pick a GGUF to start.") }
        var tokPerSec by remember { mutableStateOf(0f) }
        var kvUsed by remember { mutableStateOf(0) }
        var generating by remember { mutableStateOf(false) }
        var copying by remember { mutableStateOf(false) }
        var generationJob by remember { mutableStateOf<Job?>(null) }

        // Live graph samples — each series reset at start of each Generate
        val tokSamples = remember { mutableStateListOf<Float>() }
        val ramSamples = remember { mutableStateListOf<Float>() }
        val cpuSamples = remember { mutableStateListOf<Float>() }
        val gpuSamples = remember { mutableStateListOf<Float>() }
        // Set to true once GpuSampler confirms unavailability
        var gpuUnavailable by remember { mutableStateOf(false) }

        // Backend state
        var availableBackends by remember { mutableStateOf<List<BackendInfo>>(emptyList()) }
        var activeBackendStr by remember { mutableStateOf("") }
        var useGpu by remember { mutableStateOf(true) }
        val nGpuLayers by derivedStateOf { if (useGpu) 99 else 0 }
        val nCtx = 4096

        // Live metric snapshots for stats strip
        var currentRamMb by remember { mutableStateOf<Float?>(null) }
        var currentCpuPct by remember { mutableStateOf<Float?>(null) }

        // Graphs collapsed by default
        var showGraphs by remember { mutableStateOf(false) }

        val outputScrollState = rememberScrollState()

        LaunchedEffect(Unit) {
            val backends = withContext(Dispatchers.IO) {
                try { engine.availableBackends() } catch (e: Exception) { emptyList() }
            }
            availableBackends = backends
            android.util.Log.i("MainActivity", "Available backends: $backends")
        }

        val picker = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.OpenDocument()
        ) { uri: Uri? ->
            if (uri == null) return@rememberLauncherForActivityResult

            copying = true
            status = "Copying GGUF to cache… (may take a while for large files)"
            modelLoaded = false
            activeBackendStr = ""

            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val dest = File(cacheDir, "model.gguf")
                    contentResolver.openInputStream(uri)!!.use { input ->
                        dest.outputStream().use { output -> input.copyTo(output) }
                    }
                    localPath = dest.absolutePath
                    withContext(Dispatchers.Main) {
                        status = "Copied → ${dest.name} (${dest.length() / 1_048_576} MB). Ready."
                        copying = false
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        status = "Copy failed: ${e.message}"
                        copying = false
                    }
                }
            }
        }

        val hasGpuBackend = availableBackends.any { it.type == "gpu" || it.type == "igpu" }

        // Build stats strip text
        val statsText = buildString {
            if (tokPerSec > 0f) append("%.1f tok/s".format(tokPerSec))
            currentRamMb?.let { ram ->
                if (isNotEmpty()) append(" · ")
                append("%.0f MB".format(ram))
            }
            currentCpuPct?.let { cpu ->
                if (isNotEmpty()) append(" · ")
                append("%.0f%% CPU".format(cpu))
            }
            if (activeBackendStr.isNotEmpty()) {
                if (isNotEmpty()) append(" · ")
                append("⚡ $activeBackendStr")
            }
        }

        MaterialTheme {
            Surface(modifier = Modifier.fillMaxSize()) {
                // Root column: NO verticalScroll — output takes weight(1f) + own scroll
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    // ── Title ────────────────────────────────────────────────
                    Text(
                        "LlamaKt Bench",
                        fontSize = 20.sp,
                        style = MaterialTheme.typography.titleLarge,
                    )

                    // ── Pick GGUF + model name ────────────────────────────────
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Button(
                            onClick = { picker.launch(arrayOf("*/*")) },
                            enabled = !copying && !generating,
                            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp),
                        ) { Text("Pick GGUF") }

                        if (copying) {
                            CircularProgressIndicator(modifier = Modifier.size(20.dp))
                        }

                        localPath?.let { path ->
                            Text(
                                File(path).name,
                                style = MaterialTheme.typography.bodySmall,
                                maxLines = 1,
                                modifier = Modifier.weight(1f),
                            )
                        }
                    }

                    // ── Segmented backend selector (ONLY backend control) ──────
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        FilterChip(
                            selected = !useGpu,
                            onClick = {
                                if (!generating) {
                                    useGpu = false
                                    modelLoaded = false
                                    activeBackendStr = ""
                                }
                            },
                            label = { Text("CPU") },
                            enabled = !generating && !copying,
                        )
                        FilterChip(
                            selected = useGpu,
                            onClick = {
                                if (!generating && hasGpuBackend) {
                                    useGpu = true
                                    modelLoaded = false
                                    activeBackendStr = ""
                                }
                            },
                            label = { Text("GPU") },
                            enabled = !generating && !copying && hasGpuBackend,
                        )
                        if (!hasGpuBackend && availableBackends.isNotEmpty()) {
                            Text(
                                "(no GPU device)",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.outlineVariant,
                            )
                        }
                    }

                    // ── Preset chips ──────────────────────────────────────────
                    LazyRow(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        contentPadding = PaddingValues(horizontal = 2.dp),
                    ) {
                        items(PRESET_PROMPTS) { preset ->
                            SuggestionChip(
                                onClick = { prompt = preset },
                                label = {
                                    Text(
                                        text = preset.take(32) + if (preset.length > 32) "…" else "",
                                        maxLines = 1,
                                        style = MaterialTheme.typography.labelSmall,
                                    )
                                },
                                enabled = !generating,
                            )
                        }
                    }

                    // ── Prompt field ──────────────────────────────────────────
                    OutlinedTextField(
                        value = prompt,
                        onValueChange = { prompt = it },
                        label = { Text("Prompt") },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !generating,
                    )

                    // ── Generate / Stop ───────────────────────────────────────
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = {
                                val path = localPath ?: run {
                                    status = "No model loaded. Pick a GGUF first."
                                    return@Button
                                }
                                output = ""
                                tokPerSec = 0f
                                kvUsed = 0
                                currentRamMb = null
                                currentCpuPct = null
                                tokSamples.clear()
                                ramSamples.clear()
                                cpuSamples.clear()
                                gpuSamples.clear()
                                CpuSampler.reset()
                                generating = true
                                status = "Generating…"

                                generationJob = lifecycleScope.launch {
                                    try {
                                        if (!modelLoaded) {
                                            status = "Loading model (nGpuLayers=$nGpuLayers)…"
                                            withContext(Dispatchers.IO) {
                                                engine.load(path, nGpuLayers, nCtx)
                                            }
                                            modelLoaded = true
                                            activeBackendStr = withContext(Dispatchers.IO) {
                                                engine.activeBackend()
                                            }
                                            status = "Model loaded [$activeBackendStr]. Generating…"
                                        }

                                        var tokenCount = 0
                                        val startMs = System.currentTimeMillis()

                                        val windowSize = 16
                                        val tokenTimestamps = ArrayDeque<Long>(windowSize + 1)

                                        var lastSampleMs = 0L

                                        val messages = listOf(ChatMessage("user", prompt))
                                        engine.chat(messages).collect { token ->
                                            output += token
                                            tokenCount++

                                            val nowMs = System.currentTimeMillis()

                                            tokenTimestamps.addLast(nowMs)
                                            if (tokenTimestamps.size > windowSize) tokenTimestamps.removeFirst()

                                            val elapsedSec = (nowMs - startMs) / 1000f
                                            if (elapsedSec > 0f) tokPerSec = tokenCount / elapsedSec

                                            if (tokenTimestamps.size >= 2) {
                                                val windowSec = (tokenTimestamps.last() - tokenTimestamps.first()) / 1000f
                                                if (windowSec > 0f) {
                                                    val instantTokPerSec = (tokenTimestamps.size - 1) / windowSec

                                                    if (nowMs - lastSampleMs >= 400L) {
                                                        tokSamples.add(instantTokPerSec)

                                                        // Sample RAM + CPU + GPU on IO dispatcher
                                                        withContext(Dispatchers.IO) {
                                                            readRamMb()?.let { ramMb ->
                                                                withContext(Dispatchers.Main) {
                                                                    ramSamples.add(ramMb)
                                                                    currentRamMb = ramMb
                                                                }
                                                            }
                                                            CpuSampler.sample()?.let { cpu ->
                                                                withContext(Dispatchers.Main) {
                                                                    cpuSamples.add(cpu)
                                                                    currentCpuPct = cpu
                                                                }
                                                            }
                                                            val gpu = GpuSampler.sample()
                                                            withContext(Dispatchers.Main) {
                                                                if (GpuSampler.isUnavailable()) {
                                                                    gpuUnavailable = true
                                                                } else if (gpu != null) {
                                                                    gpuSamples.add(gpu)
                                                                }
                                                            }
                                                        }

                                                        lastSampleMs = nowMs
                                                    }
                                                }
                                            }

                                            if (tokenCount % 10 == 0) {
                                                kvUsed = engine.kvUsed()
                                            }
                                        }

                                        val elapsedSec = (System.currentTimeMillis() - startMs) / 1000f
                                        if (elapsedSec > 0f) tokPerSec = tokenCount / elapsedSec
                                        kvUsed = engine.kvUsed()
                                        status = "Done — $tokenCount tokens in %.1fs (%.1f tok/s) [$activeBackendStr]"
                                            .format(elapsedSec, tokPerSec)

                                    } catch (e: CancellationException) {
                                        status = "Interrupted."
                                        throw e
                                    } catch (e: Exception) {
                                        status = "Error: ${e.message}"
                                        modelLoaded = false
                                    } finally {
                                        generating = false
                                    }
                                }
                            },
                            enabled = !generating && !copying && localPath != null,
                        ) { Text("Generate") }

                        Button(
                            onClick = {
                                engine.interrupt()
                                generationJob?.cancel()
                                status = "Interrupted."
                                generating = false
                            },
                            enabled = generating,
                        ) { Text("Stop") }
                    }

                    // ── Stats strip ───────────────────────────────────────────
                    // Single compact line: tok/s · RAM MB · CPU% · ⚡ device
                    if (statsText.isNotEmpty()) {
                        Text(
                            statsText,
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }

                    // Status line
                    Text(
                        status,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.secondary,
                    )

                    // ── Graphs toggle ─────────────────────────────────────────
                    TextButton(
                        onClick = { showGraphs = !showGraphs },
                        contentPadding = PaddingValues(horizontal = 0.dp, vertical = 2.dp),
                    ) {
                        Text(
                            if (showGraphs) "▾ Graphs" else "▸ Graphs",
                            style = MaterialTheme.typography.labelMedium,
                        )
                    }

                    // ── Live graphs (collapsible) ─────────────────────────────
                    if (showGraphs) {
                        val graphHeight = 80.dp
                        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            LiveGraph(
                                samples = tokSamples,
                                label = "Tok/s",
                                unit = "tok/s",
                                primaryColor = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.fillMaxWidth().height(graphHeight),
                            )
                            LiveGraph(
                                samples = ramSamples,
                                label = "RAM (MB)",
                                unit = "MB",
                                primaryColor = Color(0xFF43A047),
                                modifier = Modifier.fillMaxWidth().height(graphHeight),
                            )
                            LiveGraph(
                                samples = cpuSamples,
                                label = "CPU (%)",
                                unit = "%",
                                primaryColor = Color(0xFFFF8F00),
                                modifier = Modifier.fillMaxWidth().height(graphHeight),
                            )
                            LiveGraph(
                                samples = gpuSamples,
                                label = "GPU (%)",
                                unit = "%",
                                primaryColor = Color(0xFFAB47BC),
                                modifier = Modifier.fillMaxWidth().height(graphHeight),
                                unavailable = gpuUnavailable,
                            )
                        }
                    }

                    HorizontalDivider()

                    // ── Output — weight(1f) + own verticalScroll ──────────────
                    // Always visible, never cropped, scrollable independently.
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(1f)
                            .verticalScroll(outputScrollState),
                    ) {
                        Text(
                            text = output.ifEmpty { "(output will appear here)" },
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }
            }
        }
    }
}
