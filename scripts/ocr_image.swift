import Foundation
import Vision
import AppKit

guard CommandLine.arguments.count >= 2 else {
    fputs("Usage: ocr_image.swift <image-path>\n", stderr)
    exit(2)
}

let imagePath = CommandLine.arguments[1]
let url = URL(fileURLWithPath: imagePath)

guard let image = NSImage(contentsOf: url),
      let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let cgImage = bitmap.cgImage else {
    fputs("Failed to load image: \(imagePath)\n", stderr)
    exit(1)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = ["zh-Hans", "en-US"]

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try handler.perform([request])

let lines = (request.results ?? [])
    .compactMap { $0.topCandidates(1).first?.string }
    .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

print(lines.joined(separator: "\n"))

