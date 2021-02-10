// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

struct BufferSet {
}

class SwiftnetLayer {
    enum LayerType { case fullyConnected, pooling }
    let layerType: LayerType

    let activation: BNNSActivation
    let poolingFunction: BNNSPoolingFunction!

    let height: Int
    let width: Int

    class Counts { var cBiases = 0, cInputs = 0, cOutputs = 0, cWeights = 0 }
    var counts = Counts()

    var bnnsFilter: BNNSFilter!

    var inputBuffer: UnsafeMutableRawPointer!
    var outputBuffer: UnsafeMutableRawPointer!

    init(
        activation: BNNSActivation, poolingFunction: BNNSPoolingFunction,
        kernelWidth: Int, height: Int
    ) {
        self.layerType = .pooling
        self.activation = activation
        self.poolingFunction = poolingFunction

        self.height = height
        self.width = kernelWidth

        self.counts.cBiases = height * width
        self.counts.cInputs = height * width
        self.counts.cOutputs = height * width
    }

    init(activation: BNNSActivation, cInputs: Int, cOutputs: Int) {
        self.layerType = .fullyConnected
        self.activation = activation
        self.poolingFunction = nil

        self.height = 1
        self.width = cInputs * cOutputs

        self.counts.cBiases = cOutputs
        self.counts.cInputs = cInputs
        self.counts.cOutputs = cOutputs
        self.counts.cWeights = cInputs * cOutputs
    }

    func activate() { BNNSFilterApply(bnnsFilter, inputBuffer, outputBuffer) }

    func makeFilter(
        _ pBiases: UnsafeMutableRawPointer,
        _ pInputs: UnsafeMutableRawPointer,
        _ pOutputs: UnsafeMutableRawPointer,
        _ pWeights: UnsafeMutableRawPointer
    ) {
        inputBuffer = pInputs
        outputBuffer = pOutputs
        bnnsFilter = SwiftnetFilter.makeFilter(self, pBiases, pInputs, pOutputs, pWeights)
    }

    static func getStackCounts(_ layers: [SwiftnetLayer]) -> Counts {
        let totals = Counts()

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
}
