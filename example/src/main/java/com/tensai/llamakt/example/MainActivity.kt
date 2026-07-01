package com.tensai.llamakt.example

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import com.tensai.llamakt.BackendInfo
import com.tensai.llamakt.ChatMessage
import com.tensai.llamakt.LlamaEngine
import com.tensai.llamakt.chat
import kotlinx.coroutines.*
import java.io.File

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

        // Backend state
        var availableBackends by remember { mutableStateOf<List<BackendInfo>>(emptyList()) }
        var activeBackendStr by remember { mutableStateOf("") }
        // Backend selector: true = GPU offload (nGpuLayers=99), false = CPU (nGpuLayers=0)
        var useGpu by remember { mutableStateOf(true) }
        val nGpuLayers by derivedStateOf { if (useGpu) 99 else 0 }
        val nCtx = 4096

        val scrollState = rememberScrollState()

        // List backends at startup
        LaunchedEffect(Unit) {
            val backends = withContext(Dispatchers.IO) {
                try { engine.availableBackends() } catch (e: Exception) { emptyList() }
            }
            availableBackends = backends
            android.util.Log.i("MainActivity", "Available backends: $backends")
        }

        // SAF picker: OpenDocument for any binary/mime
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
                    // Copy content:// URI into cacheDir so llama.cpp can open it by path
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

        MaterialTheme {
            Surface(modifier = Modifier.fillMaxSize()) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text("LlamaKt Bench", fontSize = 20.sp, style = MaterialTheme.typography.titleLarge)

                    // Available backends list
                    if (availableBackends.isNotEmpty()) {
                        Text("Backends:", style = MaterialTheme.typography.labelSmall)
                        availableBackends.forEach { b ->
                            val icon = when (b.type) {
                                "gpu", "igpu" -> "GPU"
                                "cpu" -> "CPU"
                                else -> b.type.uppercase()
                            }
                            Text(
                                "  [$icon] ${b.description}: ${b.name}",
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                    }

                    // Backend selector: CPU vs GPU
                    val hasGpuBackend = availableBackends.any { it.type == "gpu" || it.type == "igpu" }
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text("Backend:", style = MaterialTheme.typography.bodyMedium)
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
                        if (!hasGpuBackend) {
                            Text("(no GPU device)", style = MaterialTheme.typography.labelSmall)
                        }
                    }

                    // Active backend badge — shown after model load
                    if (activeBackendStr.isNotEmpty()) {
                        Text(
                            "Backend: $activeBackendStr",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }

                    // Model picker
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = { picker.launch(arrayOf("*/*")) },
                            enabled = !copying && !generating
                        ) { Text("Pick GGUF") }

                        if (copying) {
                            CircularProgressIndicator(modifier = Modifier.size(24.dp).align(Alignment.CenterVertically))
                        }
                    }

                    localPath?.let {
                        Text("Model: ${File(it).name}", style = MaterialTheme.typography.bodySmall)
                    }

                    // Prompt input
                    OutlinedTextField(
                        value = prompt,
                        onValueChange = { prompt = it },
                        label = { Text("Prompt") },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !generating
                    )

                    // Generate / Stop
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
                                generating = true
                                status = "Generating…"

                                generationJob = lifecycleScope.launch {
                                    try {
                                        // Load model if not already loaded (or backend changed)
                                        if (!modelLoaded) {
                                            status = "Loading model (nGpuLayers=$nGpuLayers)…"
                                            withContext(Dispatchers.IO) {
                                                engine.load(path, nGpuLayers, nCtx)
                                            }
                                            modelLoaded = true
                                            // Query active backend right after load
                                            activeBackendStr = withContext(Dispatchers.IO) {
                                                engine.activeBackend()
                                            }
                                            status = "Model loaded [$activeBackendStr]. Generating…"
                                        }

                                        var tokenCount = 0
                                        val startMs = System.currentTimeMillis()

                                        val messages = listOf(ChatMessage("user", prompt))
                                        engine.chat(messages).collect { token ->
                                            output += token
                                            tokenCount++

                                            val elapsedSec = (System.currentTimeMillis() - startMs) / 1000f
                                            if (elapsedSec > 0f) tokPerSec = tokenCount / elapsedSec

                                            // Sample KV usage every 10 tokens (avoid JNI spam)
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
                                        modelLoaded = false // force reload on next run
                                    } finally {
                                        generating = false
                                    }
                                }
                            },
                            enabled = !generating && !copying && localPath != null
                        ) { Text("Generate") }

                        Button(
                            onClick = {
                                engine.interrupt()
                                generationJob?.cancel()
                                status = "Interrupted."
                                generating = false
                            },
                            enabled = generating
                        ) { Text("Stop") }
                    }

                    // Stats bar
                    if (tokPerSec > 0f || kvUsed > 0) {
                        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                            Text("%.1f tok/s".format(tokPerSec), style = MaterialTheme.typography.labelMedium)
                            Text("KV used: $kvUsed tokens", style = MaterialTheme.typography.labelMedium)
                            if (activeBackendStr.isNotEmpty()) {
                                Text(activeBackendStr, style = MaterialTheme.typography.labelMedium)
                            }
                        }
                    }

                    // Status line
                    Text(status, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)

                    // Output
                    HorizontalDivider()
                    Text(
                        text = output.ifEmpty { "(output will appear here)" },
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(1f)
                            .verticalScroll(scrollState),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }
    }
}
