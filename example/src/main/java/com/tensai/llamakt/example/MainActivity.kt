package com.tensai.llamakt.example

import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
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
import com.tensai.llamakt.LoadProgressCallback
import com.tensai.llamakt.SamplingParams
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

// Short-output prompts designed to make sampling effects visible:
// run the same prompt several times and compare outputs across runs.
private val SAMPLING_TEST_PROMPTS = listOf(
    "Name one random animal. Answer with a single word only.",
    "Give me a random 4-digit number. Answer with the number only.",
    "Write the first sentence of a story about a lighthouse.",
    "List 5 random words, comma-separated.",
    "Complete this sentence in 10 words or less: The door opened and",
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
     * Generic live metric graph, reused for tok/s, RAM, CPU.
     *
     * @param samples      list of sampled float values (mutableStateListOf)
     * @param label        e.g. "RAM (MB)"
     * @param unit         e.g. "MB" — shown in the placeholder text
     * @param primaryColor line / grid color
     */
    @Composable
    fun LiveGraph(
        samples: List<Float>,
        label: String,
        unit: String,
        primaryColor: Color,
        modifier: Modifier = Modifier,
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
                if (current != null && avg != null) {
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

    /** Labeled slider row for a sampling parameter: label — slider — value. */
    @Composable
    fun SamplingSlider(
        label: String,
        value: Float,
        range: ClosedFloatingPointRange<Float>,
        enabled: Boolean,
        isInt: Boolean = false,
        onChange: (Float) -> Unit,
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
                modifier = Modifier.width(72.dp),
            )
            Slider(
                value = value,
                onValueChange = onChange,
                valueRange = range,
                enabled = enabled,
                modifier = Modifier.weight(1f),
            )
            Text(
                if (isInt) "${value.toInt()}" else "%.2f".format(value),
                style = MaterialTheme.typography.labelSmall,
                modifier = Modifier.width(44.dp),
            )
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun BenchScreen() {
        // State
        var localPath by remember { mutableStateOf<String?>(null) }
        var modelDisplayName by remember { mutableStateOf<String?>(null) }
        var modelMeta by remember { mutableStateOf<com.tensai.llamakt.GgufMetadata?>(null) }
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

        // Backend state
        var availableBackends by remember { mutableStateOf<List<BackendInfo>>(emptyList()) }
        var activeBackendStr by remember { mutableStateOf("") }
        var useGpu by remember { mutableStateOf(true) }
        // -1 = derive from useGpu; >= 0 = explicit layer count (autorun extra "ngl")
        var nglOverride by remember { mutableStateOf(-1) }
        val nGpuLayers by derivedStateOf { if (nglOverride >= 0) nglOverride else if (useGpu) 99 else 0 }

        // Context window. KV cache grows linearly with it (f16, Llama-3B class:
        // ~450MB at 4096). Load-time param — changing forces reload.
        var nCtx by remember { mutableStateOf(4096) }

        // CPU threads: 0 = llama.cpp auto. Load-time param — changing forces reload.
        var nThreads by remember { mutableStateOf(0) }

        // KV cache type: null = f16. Settable via autorun extra "kv" (e.g. "q8_0").
        var kvCacheType by remember { mutableStateOf<String?>(null) }

        // Flash attention: null = auto. Settable via autorun extra "fa" ("on"/"off").
        var flashAttn by remember { mutableStateOf<String?>(null) }

        // Live metric snapshots for stats strip
        var currentRamMb by remember { mutableStateOf<Float?>(null) }
        var currentCpuPct by remember { mutableStateOf<Float?>(null) }

        // Graphs collapsed by default
        var showGraphs by remember { mutableStateOf(false) }

        // Sampling controls — defaults match llama.cpp / SamplingParams
        var showSampling by remember { mutableStateOf(false) }
        var maxTokens by remember { mutableStateOf(512f) }
        var temperature by remember { mutableStateOf(0.8f) }
        var topK by remember { mutableStateOf(40f) }
        var topP by remember { mutableStateOf(0.95f) }
        var minP by remember { mutableStateOf(0.05f) }
        // Comma-separated in the UI field; also settable via autorun extra "stops"
        var stopsField by remember { mutableStateOf("") }
        var stopSequences by remember { mutableStateOf<List<String>>(emptyList()) }

        // Session persistence test hooks (autorun extras "save"/"load")
        var autorunSaveSession by remember { mutableStateOf(false) }
        var autorunLoadSession by remember { mutableStateOf(false) }

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

            // Resolve display name from SAF before kicking off the copy
            val displayName: String = contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val col = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (col >= 0 && cursor.moveToFirst()) cursor.getString(col) else null
            } ?: uri.lastPathSegment ?: "model.gguf"

            copying = true
            status = "Copying GGUF to cache… (may take a while for large files)"
            modelLoaded = false
            modelDisplayName = displayName
            activeBackendStr = ""

            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val dest = File(cacheDir, "model.gguf")
                    contentResolver.openInputStream(uri)!!.use { input ->
                        dest.outputStream().use { output -> input.copyTo(output) }
                    }
                    localPath = dest.absolutePath
                    val meta = LlamaEngine.readMetadata(dest.absolutePath)
                    withContext(Dispatchers.Main) {
                        modelMeta = meta
                        status = if (meta != null) "Ready."
                                 else "Copied → $displayName (${dest.length() / 1_048_576} MB). Ready (not a readable GGUF?)."
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

        // Build stats strip text: tok/s · KV n · RAM MB · CPU% · ⚡backend
        val statsText = buildString {
            if (tokPerSec > 0f) append("%.1f tok/s".format(tokPerSec))
            if (kvUsed > 0) {
                if (isNotEmpty()) append(" · ")
                append("KV $kvUsed")
            }
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
                // Root column: single global verticalScroll — no nested scroll
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(rememberScrollState())
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

                        modelDisplayName?.let { name ->
                            Text(
                                name,
                                style = MaterialTheme.typography.bodySmall,
                                maxLines = 1,
                                modifier = Modifier.weight(1f),
                            )
                        }
                    }

                    // ── Persistent model card (survives generation) ───────────
                    modelMeta?.let { m ->
                        Text(
                            "%s · %s · %.2fB params · ctx %d · %d layers · %d MB".format(
                                m.architecture, m.quantLabel, m.paramCount / 1e9,
                                m.contextLength, m.blockCount, m.fileSizeBytes / 1_048_576,
                            ),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.tertiary,
                        )
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

                    // ── CPU threads selector (load-time — reload on change) ───
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Text("Threads", style = MaterialTheme.typography.labelSmall)
                        listOf(0, 2, 4, 6, 8).forEach { t ->
                            FilterChip(
                                selected = nThreads == t,
                                onClick = {
                                    if (!generating) {
                                        nThreads = t
                                        modelLoaded = false
                                        activeBackendStr = ""
                                    }
                                },
                                label = { Text(if (t == 0) "auto" else "$t") },
                                enabled = !generating && !copying,
                            )
                        }
                    }

                    // ── Context size selector (load-time — reload on change) ──
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Text("Context", style = MaterialTheme.typography.labelSmall)
                        listOf(2048, 4096, 8192, 16384).forEach { c ->
                            FilterChip(
                                selected = nCtx == c,
                                onClick = {
                                    if (!generating) {
                                        nCtx = c
                                        modelLoaded = false
                                        activeBackendStr = ""
                                    }
                                },
                                label = { Text(if (c >= 1024) "${c / 1024}k" else "$c") },
                                enabled = !generating && !copying,
                            )
                        }
                    }

                    // ── KV cache type selector (load-time — reload on change) ─
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Text("KV cache", style = MaterialTheme.typography.labelSmall)
                        listOf(null, "q8_0").forEach { t ->
                            FilterChip(
                                selected = kvCacheType == t,
                                onClick = {
                                    if (!generating) {
                                        kvCacheType = t
                                        modelLoaded = false
                                        activeBackendStr = ""
                                    }
                                },
                                label = { Text(t ?: "f16") },
                                enabled = !generating && !copying,
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

                    // ── Sampling-test chips (short outputs, easy to compare) ──
                    LazyRow(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        contentPadding = PaddingValues(horizontal = 2.dp),
                    ) {
                        items(SAMPLING_TEST_PROMPTS) { preset ->
                            SuggestionChip(
                                onClick = { prompt = preset },
                                label = {
                                    Text(
                                        text = "🎲 " + preset.take(30) + if (preset.length > 30) "…" else "",
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

                    // ── Sampling controls (collapsible) ───────────────────────
                    TextButton(
                        onClick = { showSampling = !showSampling },
                        contentPadding = PaddingValues(horizontal = 0.dp, vertical = 2.dp),
                    ) {
                        Text(
                            (if (showSampling) "▾ Sampling" else "▸ Sampling") +
                                "  ·  ${maxTokens.toInt()} tok · T %.2f · K ${topK.toInt()} · P %.2f"
                                    .format(temperature, topP),
                            style = MaterialTheme.typography.labelMedium,
                        )
                    }
                    if (showSampling) {
                        Column(verticalArrangement = Arrangement.spacedBy(0.dp)) {
                            SamplingSlider(
                                label = "Max tok",
                                value = maxTokens,
                                range = 16f..4096f,
                                enabled = !generating,
                                isInt = true,
                            ) { maxTokens = it }
                            SamplingSlider(
                                label = "Temp",
                                value = temperature,
                                range = 0f..2f,
                                enabled = !generating,
                            ) { temperature = it }
                            SamplingSlider(
                                label = "Top-K",
                                value = topK,
                                range = 0f..100f,
                                enabled = !generating,
                                isInt = true,
                            ) { topK = it }
                            SamplingSlider(
                                label = "Top-P",
                                value = topP,
                                range = 0f..1f,
                                enabled = !generating,
                            ) { topP = it }
                            SamplingSlider(
                                label = "Min-P",
                                value = minP,
                                range = 0f..0.5f,
                                enabled = !generating,
                            ) { minP = it }
                            OutlinedTextField(
                                value = stopsField,
                                onValueChange = {
                                    stopsField = it
                                    stopSequences = it.split(',')
                                        .map(String::trim)
                                        .filter(String::isNotEmpty)
                                },
                                label = { Text("Stop sequences (comma-separated)") },
                                modifier = Modifier.fillMaxWidth(),
                                enabled = !generating,
                                textStyle = MaterialTheme.typography.bodySmall,
                            )
                        }
                    }

                    // Generation logic, shared by the Generate button and the
                    // adb autorun path (see LaunchedEffect below).
                    val startGeneration: () -> Unit = startGeneration@{
                                val path = localPath ?: run {
                                    status = "No model loaded. Pick a GGUF first."
                                    return@startGeneration
                                }
                                output = ""
                                tokPerSec = 0f
                                kvUsed = 0
                                currentRamMb = null
                                currentCpuPct = null
                                tokSamples.clear()
                                ramSamples.clear()
                                cpuSamples.clear()
                                CpuSampler.reset()
                                generating = true
                                status = "Generating…"

                                generationJob = lifecycleScope.launch {
                                    try {
                                        if (!modelLoaded) {
                                            status = "Loading model (nGpuLayers=$nGpuLayers, threads=${if (nThreads == 0) "auto" else nThreads})…"
                                            withContext(Dispatchers.IO) {
                                                engine.load(path, nGpuLayers, nCtx, nThreads,
                                                    LoadProgressCallback { p ->
                                                        status = "Loading model… ${(p * 100).toInt()}%"
                                                        true
                                                    },
                                                    kvCacheType, flashAttn)
                                            }
                                            modelLoaded = true
                                            activeBackendStr = withContext(Dispatchers.IO) {
                                                engine.activeBackend()
                                            }
                                            status = "Model loaded [$activeBackendStr]. Generating…"
                                        }

                                        if (autorunLoadSession) {
                                            val n = withContext(Dispatchers.IO) {
                                                engine.loadSession(File(cacheDir, "session.bin").absolutePath)
                                            }
                                            android.util.Log.i("LlamaKtBench", "session loaded: $n tokens, kvUsed=${engine.kvUsed()}")
                                        }

                                        var tokenCount = 0
                                        var firstTokenMs = 0L
                                        val startMs = System.currentTimeMillis()

                                        val windowSize = 16
                                        val tokenTimestamps = ArrayDeque<Long>(windowSize + 1)

                                        var lastSampleMs = 0L

                                        val messages = listOf(ChatMessage("user", prompt))
                                        val sampling = SamplingParams(
                                            nPredict = maxTokens.toInt(),
                                            temperature = temperature,
                                            topK = topK.toInt(),
                                            topP = topP,
                                            minP = minP,
                                            stopSequences = stopSequences,
                                        )
                                        engine.chat(messages, sampling).collect { token ->
                                            output += token
                                            tokenCount++
                                            if (firstTokenMs == 0L) firstTokenMs = System.currentTimeMillis()

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

                                                        // Sample RAM + CPU on IO dispatcher
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
                                        val ttftMs = if (firstTokenMs > 0L) firstTokenMs - startMs else -1L
                                        // Prompt size = KV used minus generated tokens; prefill
                                        // speed = prompt tokens / time to first token. Decode
                                        // speed excludes the prefill window entirely.
                                        val promptTokens = (kvUsed - tokenCount).coerceAtLeast(0)
                                        val prefillTokS = if (ttftMs > 0) promptTokens * 1000f / ttftMs else 0f
                                        val decodeTokS = if (elapsedSec > ttftMs / 1000f && tokenCount > 1)
                                            (tokenCount - 1) / (elapsedSec - ttftMs / 1000f) else 0f
                                        android.util.Log.i(
                                            "LlamaKtBench",
                                            "bench: threads=${if (nThreads == 0) "auto" else nThreads} ngl=$nGpuLayers backend=$activeBackendStr " +
                                                "kv=${kvCacheType ?: "f16"} fa=${flashAttn ?: "auto"} " +
                                                "pp=$promptTokens pp_tok/s=%.1f ttft=${ttftMs}ms tokens=$tokenCount tg_tok/s=%.2f elapsed=%.1fs model=$modelDisplayName"
                                                    .format(prefillTokS, decodeTokS, elapsedSec),
                                        )

                                        if (autorunSaveSession) {
                                            val n = withContext(Dispatchers.IO) {
                                                engine.saveSession(File(cacheDir, "session.bin").absolutePath)
                                            }
                                            android.util.Log.i("LlamaKtBench", "session saved: $n tokens, kvUsed=${engine.kvUsed()}")
                                        }

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
                    }

                    // Headless autorun for adb-driven repro loops:
                    //   adb shell am start -n com.tensai.llamakt.example/.MainActivity \
                    //     --ez autorun true --ez gpu true [--es prompt "..."]
                    // Reuses cacheDir/model.gguf from a previous pick (or a
                    // run-as push) — SAF picking is not adb-scriptable.
                    LaunchedEffect(Unit) {
                        if (intent?.getBooleanExtra("autorun", false) == true) {
                            val cached = File(cacheDir, "model.gguf")
                            if (cached.exists()) {
                                localPath = cached.absolutePath
                                modelDisplayName = "model.gguf (autorun)"
                                useGpu = intent.getBooleanExtra("gpu", true)
                                intent.getStringExtra("prompt")?.let { prompt = it }
                                intent.getStringArrayExtra("stops")?.let { stopSequences = it.toList() }
                                autorunSaveSession = intent.getBooleanExtra("save", false)
                                autorunLoadSession = intent.getBooleanExtra("load", false)
                                kvCacheType = intent.getStringExtra("kv")
                                if (intent.hasExtra("temp")) temperature = intent.getFloatExtra("temp", 0.8f)
                                if (intent.hasExtra("ngl")) nglOverride = intent.getIntExtra("ngl", -1)
                                flashAttn = intent.getStringExtra("fa")
                                android.util.Log.i("LlamaKtBench", "autorun: gpu=$useGpu model=${cached.length() / 1_048_576}MB")
                                LlamaEngine.readMetadata(cached.absolutePath)?.let { m ->
                                    modelMeta = m
                                    android.util.Log.i("LlamaKtBench", "metadata: arch=${m.architecture} name=${m.name} quant=${m.quantLabel} params=${m.paramCount} ctx=${m.contextLength} embd=${m.embeddingLength} layers=${m.blockCount} vocab=${m.vocabSize} size=${m.fileSizeBytes}")
                                }
                                startGeneration()
                            } else {
                                status = "autorun: no cached model.gguf"
                            }
                        }
                    }

                    // ── Generate / Stop ───────────────────────────────────────
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = startGeneration,
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

                    // ── Session save / load (cacheDir/session.bin) ────────────
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(
                            onClick = {
                                lifecycleScope.launch {
                                    try {
                                        val n = withContext(Dispatchers.IO) {
                                            engine.saveSession(File(cacheDir, "session.bin").absolutePath)
                                        }
                                        status = if (n >= 0) "Session saved ($n tokens)." else "Session save failed."
                                    } catch (e: Exception) {
                                        status = "Session save failed: ${e.message}"
                                    }
                                }
                            },
                            enabled = modelLoaded && !generating,
                        ) { Text("Save session") }

                        OutlinedButton(
                            onClick = {
                                val path = localPath
                                if (path == null) {
                                    status = "Pick a GGUF first."
                                } else {
                                    lifecycleScope.launch {
                                        try {
                                            if (!modelLoaded) {
                                                status = "Loading model…"
                                                withContext(Dispatchers.IO) {
                                                    engine.load(path, nGpuLayers, nCtx, nThreads,
                                                        LoadProgressCallback { p ->
                                                            status = "Loading model… ${(p * 100).toInt()}%"
                                                            true
                                                        },
                                                        kvCacheType)
                                                }
                                                modelLoaded = true
                                                activeBackendStr = withContext(Dispatchers.IO) { engine.activeBackend() }
                                            }
                                            val n = withContext(Dispatchers.IO) {
                                                engine.loadSession(File(cacheDir, "session.bin").absolutePath)
                                            }
                                            kvUsed = engine.kvUsed()
                                            status = if (n >= 0) "Session loaded ($n tokens, KV $kvUsed)."
                                                     else "Session load failed (no saved session?)."
                                        } catch (e: Exception) {
                                            status = "Session load failed: ${e.message}"
                                        }
                                    }
                                }
                            },
                            enabled = !generating && !copying && localPath != null,
                        ) { Text("Load session") }
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
                        }
                    }

                    HorizontalDivider()

                    // ── Output — plain content in the global scroll ───────────
                    Text(
                        text = output.ifEmpty { "(output will appear here)" },
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 200.dp),
                    )
                }
            }
        }
    }
}
