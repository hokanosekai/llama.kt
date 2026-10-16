package com.tensai.llamakt

import java.io.File

/**
 * Thread counts [LlamaEngine.load] hands to llama.cpp: [decode] for token generation (memory-bound,
 * the slow cores only drag it down), [batch] for prompt prefill (compute-bound, extra cores help).
 * 0 means "let llama.cpp pick".
 */
data class ThreadCounts(val decode: Int, val batch: Int)

/**
 * Picks [ThreadCounts] from the CPU topology, as one speed number per core (`cpu_capacity` when the
 * kernel exposes it, else `cpuinfo_max_freq` in kHz). Pure so it can be tested without a device.
 *
 * - decode: the big cores, [bigCoreCount]. Measured on a 2xA78+6xA55 chip, llama.cpp's own auto
 *   (all 8) was up to 6x slower than this.
 * - batch: the big cores plus half of the others, never more than every core but one. A prefill
 *   is minutes on a phone, and a thread per core starves the UI thread for all of it (Choreographer
 *   skipped ~43 frames every ~3 s on 2+6 with the 2 big cores busy). The little cores are about a
 *   third of a big one, so half of them is where the gain flattens before the sync cost shows. Not
 *   measured on a device yet: the chosen values are logged at load, bench with [LlamaEngine.load]'s
 *   explicit `nThreadsBatch` to move them.
 *   2+6 gives 5, 4+4 gives 6, 1+3+4 gives (4, 6), 4 identical cores give 4.
 */
object ThreadPolicy {

    /** With two clusters, a core within this share of the fastest one's speed counts as big. */
    private const val BIG_CORE_PERCENT = 85

    private val CPU_DIR = File("/sys/devices/system/cpu")

    /**
     * Big cores of the SoC. With three or more distinct speeds (prime + mid + little, e.g. 1+3+4)
     * that is every core above the slowest cluster: the mids decode nearly as fast as the prime.
     * With two, the faster cluster, plus any core within [BIG_CORE_PERCENT]% of the fastest.
     * 0 when [speeds] is empty.
     */
    fun bigCoreCount(speeds: List<Long>): Int {
        if (speeds.isEmpty()) return 0
        if (speeds.distinct().size >= 3) {
            val slowest = speeds.min()
            return speeds.count { it > slowest }
        }
        val max = speeds.max()
        return speeds.count { it * 100 >= max * BIG_CORE_PERCENT }
    }

    /** See the class doc. [ThreadCounts] of 0/0 when the topology is unknown. */
    fun resolve(speeds: List<Long>): ThreadCounts {
        val big = bigCoreCount(speeds)
        if (big == 0) return ThreadCounts(0, 0)
        val others = speeds.size - big
        val batch = (big + others / 2).coerceAtMost(speeds.size - 1).coerceAtLeast(big)
        return ThreadCounts(decode = big, batch = batch)
    }

    /**
     * One speed per core sysfs lets us read, empty when none: `cpu_capacity` of every core if all
     * expose it (the kernel's own normalised figure), else `cpuinfo_max_freq`.
     */
    fun readCoreSpeeds(cpuDir: File = CPU_DIR): List<Long> = try {
        val cpus = cpuDir.listFiles { f -> f.name.matches(Regex("cpu\\d+")) }.orEmpty()
        fun read(name: String) = cpus.mapNotNull { cpu ->
            runCatching { File(cpu, name).readText().trim().toLong() }.getOrNull()
        }
        val capacity = read("cpu_capacity")
        if (capacity.isNotEmpty() && capacity.size == cpus.size) capacity else read("cpufreq/cpuinfo_max_freq")
    } catch (_: Exception) {
        emptyList()
    }
}
