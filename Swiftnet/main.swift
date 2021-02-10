// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

print("Hello, World!")

var inputLayer = SwiftnetLayer(activation: .identity, poolingFunction: .average, kernelWidth: 3, height: 3)
var hiddenLayers = [SwiftnetLayer(activation: .identity, cInputs: 9, cOutputs: 9)]
var outputLayer = SwiftnetLayer(activation: .identity, cInputs: 9, cOutputs: 1)

var allLayers: [SwiftnetLayer] { [inputLayer] + hiddenLayers + [outputLayer] }

let net = Swiftnet.makeNet(layers: allLayers)

net.inputBuffer.initialize(repeating: 1)
net.pBiases.initializeMemory(as: Float.self, repeating: 0, count: net.counts.cBiases)
net.pWeights.initializeMemory(as: Float.self, repeating: 1, count: net.counts.cWeights)
net.activate()

print("in \(net.inputBuffer.map { $0 })")
//print("biases \(pBiases.map { $0 })")
//print("weights \(pWeights.map { $0 })")
print("out \(net.outputBuffer.map { $0 })")
