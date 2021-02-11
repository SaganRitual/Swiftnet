// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

print("Hello, World!")

let netDispatch = DispatchQueue(
    label: "cc.dispatch.net", target: DispatchQueue.global()
)

let callbackDispatch = DispatchQueue(
    label: "cb.dispatch.net", target: DispatchQueue.global()
)

let semaphore = DispatchSemaphore(value: 0)

class Gremlin {
    var idNumber = 0

    var inputLayer = SwiftnetLayer(activation: .identity, poolingFunction: .average, kernelWidth: 3, height: 3)
    var hiddenLayers = [SwiftnetLayer(activation: .identity, cInputs: 9, cOutputs: 1)]

    var outputLayer = SwiftnetLayer(
        activation: .identity, cInputs: 9, cOutputs: 1, calculateControls: !lastLayerIsEmpty
    )

    var allLayers: [SwiftnetLayer] { [inputLayer] + hiddenLayers }// + [outputLayer] }

    var net: Swiftnet!
    var pBiases: UnsafeMutableRawPointer!
    var pIO: UnsafeMutableRawPointer!
    var pWeights: UnsafeMutableRawPointer!

    init(idNumber: Int) {
        self.idNumber = idNumber
        let counts = Swiftnet.getStackCounts(allLayers)

        print("cb = \(counts.cBiases), cpIO = \(counts.cInputs + counts.cOutputs), cpW = \(counts.cWeights)")

        self.pBiases = UnsafeMutableRawPointer.allocate(byteCount: bytes(counts.cBiases), alignment: MemoryLayout<Float>.alignment)
        self.pIO = UnsafeMutableRawPointer.allocate(byteCount: bytes(counts.cInputs + counts.cOutputs), alignment: MemoryLayout<Float>.alignment)
        self.pWeights = UnsafeMutableRawPointer.allocate(byteCount: bytes(counts.cWeights), alignment: MemoryLayout<Float>.alignment)

        pBiases.initializeMemory(as: Float.self, repeating: 0, count: counts.cBiases)
        pWeights.initializeMemory(as: Float.self, repeating: 1, count: counts.cWeights)

        Swiftnet.toMutableBuffer(from: pIO, cElements: counts.cInputs).initialize(repeating: Float(idNumber + 1))

        self.net = Swiftnet.makeNet(
            layers: allLayers, pBiases: pBiases, pIO: pIO, pWeights: pWeights, callbackDispatch: callbackDispatch
        )
    }

    func activate() {
        print("in for \(idNumber) \(net.inputBuffer.map { $0 })")

        net.activate { [self] in
            print("out for \(idNumber) \(net.outputBuffer.map { $0 })")

            assert(net.outputBuffer.allSatisfy { !$0.isNaN && $0 != 0  })
            semaphore.signal()
        }
    }
}

let count = 100

for idNumber in 0..<count {
    netDispatch.async {
        print("started \(idNumber)")
        let g = Gremlin(idNumber: idNumber)
        g.activate()
    }
}

for _ in 0..<count { semaphore.wait() }
