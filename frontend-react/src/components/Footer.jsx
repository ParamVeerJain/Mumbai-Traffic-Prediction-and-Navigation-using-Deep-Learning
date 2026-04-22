import { Github, Twitter, Linkedin } from "lucide-react";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-content">
        <p>&copy; 2026 Mumbai SmartNav. Intelligent Routing Powered by Deep Learning.</p>
        <div className="socials">
          <Github size={20} className="social-icon" />
          <Twitter size={20} className="social-icon" />
          <Linkedin size={20} className="social-icon" />
        </div>
      </div>
    </footer>
  );
}
