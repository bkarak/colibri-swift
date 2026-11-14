import ColibriCore
import Metal
import Testing
@testable import ColibriMetal

@Suite struct MetalSkeletonTests {
    /// The MSL string compiles at runtime under plain `swift build` (no Xcode build step),
    /// and a batch round-trips through the backend.
    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func shaderCompilesAndBatchRuns() throws {
        let metal = try MetalBackend()
        _ = try metal.pipeline("colibri_copy_f32")
        let a = try metal.makeBuffer(.f32, count: 4, label: "a")
        let b = try metal.makeBuffer(.f32, count: 4, label: "b")
        for i in 0..<4 { a.slice().floats[i] = Float(i + 1) }
        metal.beginBatch(label: "copy")
        metal.copy(a.slice(), into: b.slice(), count: 4)
        try metal.synchronize()
        #expect(Array(b.slice().floats) == [1, 2, 3, 4])
    }
}
