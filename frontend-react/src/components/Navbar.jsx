import { Link, useLocation } from "react-router-dom";
import { Map } from "lucide-react";

export default function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar glass-panel">
      <div className="nav-brand">
        <Map className="brand-icon text-accent" />
        <Link to="/" className="brand-text">Mumbai SmartNav</Link>
      </div>
      <div className="nav-links">
        <Link to="/" className={location.pathname === "/" ? "active" : ""}>Home</Link>
        <Link to="/app" className={location.pathname === "/app" ? "active" : ""}>Live Traffic</Link>
        <Link to="/flood-demo" className={location.pathname === "/flood-demo" ? "active" : ""}>Flood Demo</Link>
      </div>
    </nav>
  );
}
