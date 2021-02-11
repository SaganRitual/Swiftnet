// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

class SwiftnetLayer {
    enum LayerType { case fullyConnected, pooling }
    let layerType: LayerType

    let activation: BNNSActivationFunction
    let poolingFunction: BNNSPoolingFunction!

    let cChannels: Int
    let height: Int
    let width: Int

    class Counts: CustomDebugStringConvertible {
        var debugDescription: String { "cElements: \(cBiases), \(cInputs), \(cOutputs), \(cWeights)" }
        var cBiases = 0, cInputs = 0, cOutputs = 0, cWeights = 0
    }

    var counts = Counts()

    var bnnsFilter: BNNSFilter!

    var inputBuffer: UnsafeMutableRawPointer!
    var outputBuffer: UnsafeMutableRawPointer!

    init(
        activation: BNNSActivationFunction, poolingFunction: BNNSPoolingFunction,
        kernelWidth: Int, height: Int, cChannels: Int
    ) {
        self.layerType = .pooling
        self.activation = activation
        self.poolingFunction = poolingFunction

        self.cChannels = cChannels
        self.height = height
        self.width = kernelWidth

        self.counts.cBiases = height * width
        self.counts.cInputs = height * width
        self.counts.cOutputs = height * width
    }

    init(
        activation: BNNSActivationFunction, cInputs: Int, cOutputs: Int,
        calculateControls: Bool = true
    ) {
        self.layerType = .fullyConnected
        self.activation = activation
        self.poolingFunction = nil

        self.cChannels = 1
        self.height = 1
        self.width = cInputs * cOutputs

        self.counts.cInputs = cInputs
        self.counts.cOutputs = cOutputs

        if calculateControls {
            self.counts.cBiases = cOutputs
            self.counts.cWeights = cInputs * cOutputs
        }
    }

    func activate() { BNNSFilterApply(bnnsFilter, inputBuffer, outputBuffer) }

    func makeFilter(
        _ pBiases: UnsafeMutableRawPointer?,
        _ pInputs: UnsafeMutableRawPointer,
        _ pOutputs: UnsafeMutableRawPointer,
        _ pWeights: UnsafeMutableRawPointer?
    ) {
        inputBuffer = pInputs
        outputBuffer = pOutputs

        bnnsFilter = SwiftnetFilter.makeFilter(self, pBiases, pInputs, pOutputs, pWeights)
    }
}
