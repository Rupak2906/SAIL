import { Link } from 'react-router-dom'
import { Circle } from 'lucide-react'

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 bg-white border-b border-border h-16 flex items-center px-12">
      <div className="flex items-center justify-between w-full">

        <Link to="/" className="flex items-center gap-2 no-underline">
          <Circle size={10} className="text-primary fill-primary" />
          <span className="text-primary font-semibold text-xl leading-none">
            SAIL
          </span>
        </Link>

        <div className="flex items-center gap-8">
          <a
            href="#about"
            className="text-text-secondary hover:text-text-primary text-sm font-medium no-underline transition-colors"
          >
            About
          </a>
          <a
            href="#how-it-works"
            className="text-text-secondary hover:text-text-primary text-sm font-medium no-underline transition-colors"
          >
            How it Works
          </a>
          <Link
            to="/demo"
            className="bg-primary hover:bg-primary-hover text-white text-sm font-medium px-5 py-2 rounded-lg no-underline"
          >
            Try Demo
          </Link>
        </div>

      </div>
    </nav>
  )
}
