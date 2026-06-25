import { Link } from 'react-router-dom'
import { Upload, Cpu, Eye } from 'lucide-react'

const steps = [
  {
    icon: Upload,
    title: 'Upload NIR Image',
    description: 'Drag and drop a near-infrared forearm image into the interface.',
  },
  {
    icon: Cpu,
    title: 'AI Segments Veins',
    description: 'The U-Net++ model detects and localizes subcutaneous vein structures.',
  },
  {
    icon: Eye,
    title: 'View Overlay',
    description:
      'A precise vascular map is overlaid on the original image for clinical review.',
  },
]

export default function Landing() {
  return (
    <main>

      {/* ── Section 1: Hero ── */}
      <section className="bg-white pt-24 pb-24 flex flex-col items-center text-center px-6">
        <p className="text-[13px] font-medium uppercase tracking-widest text-text-secondary">
          Research Demo · Not for Clinical Use
        </p>
        <h1 className="mt-5 text-5xl font-semibold text-text-primary max-w-[640px] leading-tight">
          Vein Detection Powered by Deep Learning
        </h1>
        <p className="mt-4 text-lg text-text-secondary max-w-[520px] leading-relaxed">
          Near-infrared imaging and U-Net++ segmentation to non-invasively map
          subcutaneous veins on forearm skin.
        </p>
        <Link
          to="/demo"
          className="mt-10 bg-primary hover:bg-primary-hover text-white font-medium text-base px-8 py-[14px] rounded-lg no-underline"
        >
          Try the Demo
        </Link>
      </section>

      {/* ── Section 2: Stats ── */}
      <section className="bg-white border-t border-b border-border py-10 px-12">
        <div className="flex items-start justify-center gap-16 flex-wrap">

          <div className="flex flex-col items-center text-center">
            <span className="text-[32px] font-semibold text-primary leading-none">
              26–40%
            </span>
            <span className="mt-2 text-sm text-text-secondary max-w-[160px] leading-snug">
              first-attempt IV failure rate in general patients
            </span>
          </div>

          <div className="flex flex-col items-center text-center">
            <span className="text-[32px] font-semibold text-primary leading-none">
              2,100+
            </span>
            <span className="mt-2 text-sm text-text-secondary max-w-[160px] leading-snug">
              NIR training images (CUBITAL + VEINCV-RL)
            </span>
          </div>

          <div className="flex flex-col items-center text-center">
            <span className="text-[32px] font-semibold text-primary leading-none">
              U-Net++
            </span>
            <span className="mt-2 text-sm text-text-secondary max-w-[160px] leading-snug">
              segmentation architecture with pretrained encoder
            </span>
          </div>

        </div>
      </section>

      {/* ── Section 3: How it Works ── */}
      <section className="bg-surface pt-20 pb-20 px-12 flex flex-col items-center text-center">
        <p className="text-[13px] font-medium uppercase tracking-widest text-primary">
          How It Works
        </p>
        <h2 className="mt-3 text-[32px] font-semibold text-text-primary mb-12 leading-tight">
          Three steps from image to insight
        </h2>

        <div className="flex flex-row gap-6 justify-center flex-wrap w-full max-w-4xl">
          {steps.map(({ icon: Icon, title, description }) => (
            <div
              key={title}
              className="bg-white border border-border rounded-xl px-6 py-8 flex flex-col items-start text-left flex-1 min-w-[220px]"
            >
              <Icon size={24} strokeWidth={1.5} className="text-primary" />
              <h3 className="mt-4 text-lg font-medium text-text-primary">
                {title}
              </h3>
              <p className="mt-2 text-[15px] text-text-secondary leading-relaxed">
                {description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Section 4: Footer ── */}
      <footer className="bg-white border-t border-border py-8 px-12 flex items-center justify-between">
        <span className="text-sm text-text-secondary">
          SAIL — AI-Powered Vein Detection
        </span>
        <span className="text-[13px] text-text-muted">
          Research project · SAIL Internship 2026
        </span>
      </footer>

    </main>
  )
}
