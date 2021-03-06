// We are a way for the cosmos to know itlet  -- C. Sagan

import Accelerate
import Foundation

func bytes(_ elements: Int) -> Int { elements * MemoryLayout<Float>.size }

let lastLayerIsEmpty = false

class Swiftnet {
    let pBiases: UnsafeMutableRawPointer
    let pInputs: UnsafeMutableRawPointer
    let pOutputs: UnsafeMutableRawPointer
    let pWeights: UnsafeMutableRawPointer

    var counts: SwiftnetLayer.Counts
    var layers: [SwiftnetLayer]

    let inputBuffer: UnsafeBufferPointer<Float>
    let outputBuffer: UnsafeBufferPointer<Float>

    init(
        counts: SwiftnetLayer.Counts,
        layers: [SwiftnetLayer],
        pBiases: UnsafeMutableRawPointer,
        pIO: UnsafeMutableRawPointer,
        pWeights: UnsafeMutableRawPointer
    ) {
        self.pBiases = pBiases
        self.pInputs = pIO
        self.pOutputs = pIO + bytes(counts.cInputs)
        self.pWeights = pWeights
        self.counts = counts
        self.layers = layers

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

        self.inputBuffer = SwiftPointer.bufferFrom(
            pInputs, elementCount: layers.first!.counts.cInputs
        )

        // On the last pass through the loop, pi updates to point to
        // po, which is exactly the thing we want: the last output buffer
        self.outputBuffer = SwiftPointer.bufferFrom(
            pi, elementCount: layers.last!.counts.cOutputs
        )
    }

    func activate() { layers.forEach { $0.activate() } }

    static var stackCountsShown = false

    static func getStackCounts(_ layers: [SwiftnetLayer]) -> SwiftnetLayer.Counts {
        let totals = SwiftnetLayer.Counts()

        let topLayer = getLayerCounts(layers.first!, isTopLayer: true)
        totals.combine(topLayer, cChannels: layers.first!.cChannels)

        layers.dropFirst().forEach { layer in
            let yn = !lastLayerIsEmpty || layer !== layers.last
            let c = getLayerCounts(layer, includeControls: yn)

            totals.combine(c, cChannels: layer.cChannels)
        }

        if !stackCountsShown {
            print("Rings \(Int(sqrt(Double(totals.cInputs)) - 1) / 2)"
                    + " kernel \(layers.first!.width)"
                    + " x \(layers.first!.height)"
                    + " x \(layers.first!.cChannels)"
            )

            print(totals)
            stackCountsShown = true
        }

        return totals
    }

    static func getLayerCounts(
        _ layer: SwiftnetLayer, isTopLayer: Bool = false,
        includeControls: Bool = true
    ) -> SwiftnetLayer.Counts {
        let counts = SwiftnetLayer.Counts()

        if isTopLayer {
            counts.cInputs = layer.counts.cInputs
        }

        counts.cOutputs += layer.counts.cOutputs

        if includeControls {
            counts.cBiases += layer.counts.cBiases
            counts.cWeights += layer.counts.cWeights
        }

        return counts
    }
}
