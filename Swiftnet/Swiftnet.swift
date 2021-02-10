// We are a way for the cosmos to know itlet  -- C. Sagan

import Accelerate
import Foundation

func bytes(_ elements: Int) -> Int { elements * MemoryLayout<Float>.size }

class Swiftnet {
    static let netDispatch = DispatchQueue(
        label: "net.dispatch.rob",
//        attributes: .concurrent,
        target: DispatchQueue.global()
    )

    let callbackDispatch: DispatchQueue

    let pBiases: UnsafeMutableRawPointer
    let pInputs: UnsafeMutableRawPointer
    let pOutputs: UnsafeMutableRawPointer
    let pWeights: UnsafeMutableRawPointer

    var counts: SwiftnetLayer.Counts
    var layers: [SwiftnetLayer]

    let inputBuffer: UnsafeBufferPointer<Float>
    let outputBuffer: UnsafeBufferPointer<Float>

    // All the other memory is owned by the creator of the net
    deinit { pOutputs.deallocate() }

    static func toMutableBuffer(
        from p: UnsafeMutableRawPointer, cElements: Int
    ) -> UnsafeMutableBufferPointer<Float> {
        let t = p.bindMemory(to: Float.self, capacity: cElements)
        return UnsafeMutableBufferPointer(start: t, count: cElements)
    }

    static func toBuffer(
        from p: UnsafeMutableRawPointer, cElements: Int
    ) -> UnsafeBufferPointer<Float> {
        let t = p.bindMemory(to: Float.self, capacity: cElements)
        return UnsafeBufferPointer(start: t, count: cElements)
    }

    private init(
        counts: SwiftnetLayer.Counts, layers: [SwiftnetLayer],
        pBiases: UnsafeMutableRawPointer,
        pInputs: UnsafeMutableRawPointer,
        pWeights: UnsafeMutableRawPointer,
        callbackDispatch: DispatchQueue
    ) {
        self.pBiases = pBiases
        self.pInputs = pInputs
        self.pWeights = pWeights
        self.counts = counts
        self.layers = layers
        self.callbackDispatch = callbackDispatch

        self.pOutputs = UnsafeMutableRawPointer.allocate(
            byteCount: bytes(counts.cInputs + counts.cOutputs),
            alignment: MemoryLayout<Float>.alignment
        )

        var pb = pBiases, pi = pInputs, po = pOutputs, pw = pWeights

        layers.forEach { layer in
            layer.makeFilter(pb, pi, po, pw)
            pb += bytes(layer.counts.cBiases)
            pi = po
            po += bytes(layer.counts.cOutputs)
            pw += bytes(layer.counts.cWeights)
        }

        self.inputBuffer = Swiftnet.toBuffer(
            from: pInputs, cElements: layers.first!.counts.cInputs
        )

        // On the last pass through the loop, pi updates to point to
        // po, which is exactly the thing we want: the last output buffer
        self.outputBuffer = Swiftnet.toBuffer(
            from: pi, cElements: layers.last!.counts.cOutputs
        )
    }

    func activate(_ onComplete: @escaping () -> Void) {
        Swiftnet.netDispatch.async { [self] in
            layers.forEach { $0.activate() }
            onComplete()
        }
    }

    static func getStackCounts(_ layers: [SwiftnetLayer]) -> SwiftnetLayer.Counts {
        let totals = SwiftnetLayer.Counts()

        layers.forEach { layer in
            if layer === layers.first {
                totals.cInputs = layer.counts.cInputs
            } else {
                totals.cOutputs += layer.counts.cOutputs
            }

            totals.cBiases += layer.counts.cBiases
            totals.cWeights += layer.counts.cWeights
        }

        return totals
    }

    static func makeNet(
        layers: [SwiftnetLayer],
        pBiases: UnsafeMutableRawPointer,
        pInputs: UnsafeMutableRawPointer,
        pWeights: UnsafeMutableRawPointer,
        callbackDispatch: DispatchQueue = DispatchQueue.main
    ) -> Swiftnet {
        let counts = getStackCounts(layers)

        return Swiftnet(
            counts: counts, layers: layers, pBiases: pBiases,
            pInputs: pInputs, pWeights: pWeights,
            callbackDispatch: callbackDispatch
        )
    }
}
