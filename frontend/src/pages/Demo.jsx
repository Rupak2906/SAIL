import { useRef, useState } from 'react'
import { Upload } from 'lucide-react'
import { predict } from '../api/predict'

const STATUS_DOT = {
  ready:      'bg-gray-400',
  processing: 'bg-amber-400 animate-pulse',
  done:       'bg-green-500',
}

const STATUS_LABEL = {
  ready:      'Ready',
  processing: 'Processing…',
  done:       'Done',
}

export default function Demo() {
  const fileInputRef = useRef(null)

  const [imageFile, setImageFile]       = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [overlayImage, setOverlayImage] = useState(null)
  const [status, setStatus]             = useState('ready')
  const [metrics, setMetrics]           = useState({ dice: null, confidence: null, time: null })

  function handleFileChange(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    setOverlayImage(null)
    setStatus('ready')
    setMetrics({ dice: null, confidence: null, time: null })
  }

  function handleDrop(e) {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (!file) return
    if (!['image/png', 'image/jpeg'].includes(file.type)) return
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    setOverlayImage(null)
    setStatus('ready')
    setMetrics({ dice: null, confidence: null, time: null })
  }

  async function handleRunDetection() {
    if (!imageFile) return
    setStatus('processing')
    const result = await predict(imageFile)
    setOverlayImage(result.overlay)
    setMetrics({
      dice:       String(result.dice_score),
      confidence: `${result.confidence}%`,
      time:       `${result.processing_time}s`,
    })
    setStatus('done')
  }

  return (
    <div className="bg-white min-h-screen py-12 px-12">
      <div className="max-w-[1100px] mx-auto">

        {/* ── Top Bar ── */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-text-primary">
              Vein Detection Demo
            </h1>
            <p className="mt-1 text-[15px] text-text-secondary">
              Upload a near-infrared forearm image to see vein segmentation
            </p>
          </div>
          <span className="mt-1 bg-[#FEF3C7] text-[#92400E] text-xs px-3 py-1 rounded-full whitespace-nowrap">
            Research use only — not for clinical decisions
          </span>
        </div>

        {/* ── Main Two-Column Area ── */}
        <div className="mt-8 grid grid-cols-2 gap-6">

          {/* Left — Upload */}
          <div className="bg-white border border-border rounded-xl p-6">
            <p className="text-[13px] font-medium text-text-secondary uppercase tracking-wider mb-4">
              Input Image
            </p>

            {/* Drop zone */}
            <div
              className="border-2 border-dashed border-border rounded-xl h-[280px] bg-surface flex items-center justify-center cursor-pointer overflow-hidden"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
            >
              {imagePreview ? (
                <img
                  src={imagePreview}
                  alt="Selected NIR image"
                  className="w-full h-full object-cover rounded-xl"
                />
              ) : (
                <div className="flex flex-col items-center gap-2 select-none">
                  <Upload size={32} className="text-text-muted" />
                  <span className="text-sm text-text-muted">Drop NIR image here</span>
                  <span className="text-sm text-primary font-medium">Browse files</span>
                </div>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg"
              className="hidden"
              onChange={handleFileChange}
            />

            <button
              onClick={handleRunDetection}
              disabled={!imageFile || status === 'processing'}
              className="mt-4 w-full bg-primary hover:bg-primary-hover text-white font-medium text-sm py-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Run Detection
            </button>
          </div>

          {/* Right — Result */}
          <div className="bg-white border border-border rounded-xl p-6">
            <p className="text-[13px] font-medium text-text-secondary uppercase tracking-wider mb-4">
              Segmentation Overlay
            </p>

            {/* Result zone */}
            <div className="h-[280px] rounded-xl bg-surface border border-border flex items-center justify-center overflow-hidden">
              {overlayImage ? (
                <img
                  src={`data:image/png;base64,${overlayImage}`}
                  alt="Segmentation overlay"
                  className="w-full h-full object-cover rounded-xl"
                  onError={(e) => { e.target.src = overlayImage }}
                />
              ) : (
                <span className="text-sm text-text-muted">Overlay will appear here</span>
              )}
            </div>

            {/* Status indicator */}
            <div className="mt-3 flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${STATUS_DOT[status]}`} />
              <span className="text-[13px] text-text-secondary">
                {STATUS_LABEL[status]}
              </span>
            </div>
          </div>
        </div>

        {/* ── Metrics Row ── */}
        <div className="mt-6 grid grid-cols-3 gap-4">

          <div className="bg-white border border-border rounded-xl px-6 py-5">
            <p className="text-[12px] font-medium text-text-secondary uppercase tracking-wider">
              Dice Score
            </p>
            <p className="mt-1 text-[28px] font-semibold text-text-primary">
              {metrics.dice ?? '—'}
            </p>
          </div>

          <div className="bg-white border border-border rounded-xl px-6 py-5">
            <p className="text-[12px] font-medium text-text-secondary uppercase tracking-wider">
              Confidence
            </p>
            <p className="mt-1 text-[28px] font-semibold text-text-primary">
              {metrics.confidence ?? '—'}
            </p>
          </div>

          <div className="bg-white border border-border rounded-xl px-6 py-5">
            <p className="text-[12px] font-medium text-text-secondary uppercase tracking-wider">
              Processing Time
            </p>
            <p className="mt-1 text-[28px] font-semibold text-text-primary">
              {metrics.time ?? '—'}
            </p>
          </div>

        </div>
      </div>
    </div>
  )
}
