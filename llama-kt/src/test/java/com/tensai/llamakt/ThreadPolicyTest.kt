package com.tensai.llamakt

import org.junit.Assert.assertEquals
import org.junit.Test

class ThreadPolicyTest {

    private fun cores(vararg groups: Pair<Int, Long>) = groups.flatMap { (n, khz) -> List(n) { khz } }

    @Test
    fun dimensity900IsTwoBigCoresDecodeAndFiveForPrefill() {
        // 2x A78 at 2.4 GHz, 6x A55 at 2.0 GHz
        val counts = ThreadPolicy.resolve(cores(2 to 2_400_000L, 6 to 2_000_000L))
        assertEquals(ThreadCounts(decode = 2, batch = 5), counts)
    }

    @Test
    fun fourPlusFour() {
        assertEquals(ThreadCounts(4, 6), ThreadPolicy.resolve(cores(4 to 2_800_000L, 4 to 1_800_000L)))
    }

    @Test
    fun identicalCoresUseThemAll() {
        assertEquals(ThreadCounts(4, 4), ThreadPolicy.resolve(cores(4 to 2_000_000L)))
    }

    @Test
    fun threeTiersCountPrimeAndMidsAsBig() {
        // 1x X1 3.0, 3x A78 2.4 (80%, still above the slowest cluster), 4x A55 1.8
        assertEquals(ThreadCounts(4, 6), ThreadPolicy.resolve(cores(1 to 3_000_000L, 3 to 2_400_000L, 4 to 1_800_000L)))
    }

    @Test
    fun midCoreWithinCutoffCountsAsBig() {
        // A78 2.6 vs X1 3.0 = 87%
        assertEquals(4, ThreadPolicy.bigCoreCount(cores(1 to 3_000_000L, 3 to 2_600_000L, 4 to 1_800_000L)))
    }

    @Test
    fun unknownTopologyLeavesLlamaToPick() {
        assertEquals(ThreadCounts(0, 0), ThreadPolicy.resolve(emptyList()))
        assertEquals(0, ThreadPolicy.bigCoreCount(emptyList()))
    }
}
