// We are a way for the cosmos to know itlet  -- C. Sagan

import Accelerate
import Foundation

func bytes(_ elements: Int) -> Int { elements * MemoryLayout<Float>.size }

class Swiftnet {
    let pBiases: UnsafeMutableRawPointer
    let pInputs: UnsafeMutableRawPointer
    let pOutputs: UnsafeMutableRawPointer
    let pWeights: UnsafeMutableRawPointer

    var counts: SwiftnetLayer.Counts
    var layers: [SwiftnetLayer]

    let inputBuffer: UnsafeMutableBufferPointer<Float>
    let outputBuffer: UnsafeMutableBufferPointer<Float>

    static func toMutableBuffer(
        from p: UnsafeMutableRawPointer, cElements: Int
    ) -> UnsafeMutableBufferPointer<Float> {
        let t = p.bindMemory(to: Float.self, capacity: cElements)
        return UnsafeMutableBufferPointer(start: t, count: cElements)
    }

    private init(
        counts: SwiftnetLayer.Counts, layers: [SwiftnetLayer],
        pBiases: UnsafeMutableRawPointer,
        pInputs: UnsafeMutableRawPointer,
        pOutputs: UnsafeMutableRawPointer,
        pWeights: UnsafeMutableRawPointer
    ) {
        self.pBiases = pBiases
        self.pInputs = pInputs
        self.pOutputs = pOutputs
        self.pWeights = pWeights
        self.counts = counts
        self.layers = layers

        var pb = pBiases, pi = pInputs, po = pOutputs, pw = pWeights

        layers.forEach { layer in
            layer.makeFilter(pb, pi, po, pw)
            pb += bytes(layer.counts.cBiases)
            pi += bytes(layer.counts.cInputs)
            po += bytes(layer.counts.cOutputs)
            pw += bytes(layer.counts.cWeights)
        }

        self.inputBuffer = Swiftnet.toMutableBuffer(
            from: pInputs, cElements: layers.first!.counts.cInputs
        )

        // On the last pass through the loop, pi updates to point to
        // po, which is exactly the thing we want: the last output buffer
        self.outputBuffer = Swiftnet.toMutableBuffer(
            from: pi, cElements: layers.last!.counts.cOutputs
        )
    }

    func activate() { layers.forEach { $0.activate() } }

    static func makeNet(layers: [SwiftnetLayer]) -> Swiftnet {
        let counts = SwiftnetLayer.getStackCounts(layers)

        let pBiases = UnsafeMutableRawPointer.allocate(
            byteCount: bytes(counts.cBiases),
            alignment: MemoryLayout<Float>.alignment
        )

        let pIO = UnsafeMutableRawPointer.allocate(
            byteCount: bytes(counts.cInputs + counts.cOutputs),
            alignment: MemoryLayout<Float>.alignment
        )

        let pOutputs = pIO + bytes(counts.cInputs)

        let pWeights = UnsafeMutableRawPointer.allocate(
            byteCount: bytes(counts.cWeights),
            alignment: MemoryLayout<Float>.alignment
        )

        return Swiftnet(
            counts: counts, layers: layers,
            pBiases: pBiases, pInputs: pIO,
            pOutputs: pOutputs, pWeights: pWeights
        )
    }
}
