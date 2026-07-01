plugins { id("com.android.library"); id("org.jetbrains.kotlin.android") }
android {
    namespace = "com.tensai.llamakt"
    compileSdk = 35
    defaultConfig {
        minSdk = 29
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { arguments += listOf("-DANDROID_STL=c++_shared") } }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
    ndkVersion = "27.2.12479018"
}
kotlin { jvmToolchain(21) }
dependencies { implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0") }
