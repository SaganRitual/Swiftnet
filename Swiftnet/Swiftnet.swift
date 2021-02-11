// We are a way for the cosmos to know itlet  -- C. Sagan

import Accelerate
import Foundation

func bytes(_ elements: Int) -> Int { elements * MemoryLayout<Float>.size }

let lastLayerIsEmpty = false

class Swiftnet {
    static let netDispatch = DispatchQueue(
        label: "net.dispatch.rob",
        attributes: [],
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
        pIO: UnsafeMutableRawPointer,
        pWeights: UnsafeMutableRawPointer,
        callbackDispatch: DispatchQueue
    ) {
        self.pBiases = pBiases
        self.pInputs = pIO
        self.pOutputs = pIO + bytes(counts.cInputs)
        self.pWeights = pWeights
        self.counts = counts
        self.layers = layers
        self.callbackDispatch = callbackDispatch

        var pb = pBiases, pi = pInputs, po = pOutputs, pw = pWeights

        layers.forEach { layer in
            if layer === layers.first {
                layer.makeFilter(pb, pi, po, nil)

                pb += bytes(layer.counts.cBiases)

            } else if !lastLayerIsEmpty || layer !== layers.last {
                layer.makeFilter(pb, pi, po, pw)

                pb += bytes(layer.counts.cBiases)
                pw += bytes(layer.counts.cWeights)
            } else {
                layer.makeFilter(nil, pi, po, nil)
            }

            pi = po
            po += bytes(layer.counts.cOutputs)
        }

        self.inputBuffer = Swiftnet.toBuffer(
            from: pInputs, cElements: layers.first!.counts.cInputs
        )

        // On the last pass through the loop, pi updates to point to
        // po, which is exactly the thing we want: the last output buffer
        self.outputBuffer = Swiftnet.toBuffer(
            from: pi, cElements: layers.last!.counts.cOutputs
        )

        print("net init in \(self.inputBuffer) / \(self.inputBuffer.count) out \(self.outputBuffer) / \(self.outputBuffer.count)")
    }

    func activate(_ onComplete: @escaping () -> Void) {
        Swiftnet.netDispatch.async { [self] in
            layers.forEach { $0.activate() }
            callbackDispatch.async(execute: onComplete)
        }
    }

    static func getStackCounts(_ layers: [SwiftnetLayer]) -> SwiftnetLayer.Counts {
        let totals = SwiftnetLayer.Counts()

        layers.forEach { layer in
            if layer === layers.first {
                totals.cInputs = layer.counts.cInputs
            }

            totals.cOutputs += layer.counts.cOutputs

            if !lastLayerIsEmpty || layer !== layers.last {
                totals.cBiases += layer.counts.cBiases
                totals.cWeights += layer.counts.cWeights
            }
        }

        print("totals \(totals)")

        return totals
    }

    static func makeNet(
        layers: [SwiftnetLayer],
        pBiases: UnsafeMutableRawPointer,
        pIO: UnsafeMutableRawPointer,
        pWeights: UnsafeMutableRawPointer,
        callbackDispatch: DispatchQueue = DispatchQueue.main
    ) -> Swiftnet {
        let counts = getStackCounts(layers)

        return Swiftnet(
            counts: counts, layers: layers, pBiases: pBiases,
            pIO: pIO, pWeights: pWeights, callbackDispatch: callbackDispatch
        )
    }
}
