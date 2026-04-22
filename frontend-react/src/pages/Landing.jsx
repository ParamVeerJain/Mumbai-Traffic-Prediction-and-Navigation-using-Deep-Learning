import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Activity, MapPinned, Zap, Clock, ShieldCheck, ChevronRight } from "lucide-react";

const features = [
  { icon: Activity, title: "Deep Learning Insights", desc: "Leveraging advanced models to predict complex traffic patterns and delays." },
  { icon: MapPinned, title: "Smart Routing", desc: "Dynamically calculates the optimal path considering predictive congestion data." },
  { icon: Zap, title: "Google Maps API Ready", desc: "Built with a standardized architecture that seamlessly processes data structures identical to real-world Google Maps Traffic APIs." },
];

export default function Landing() {
  return (
    <div className="landing-page">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-glow"></div>
        <motion.div 
          className="hero-content"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <motion.div 
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="badge"
          >
            <Zap size={14} className="text-accent" /> Major Project 2.0
          </motion.div>
          <h1 className="hero-title">Navigate Mumbai <br/> Like Never <span className="gradient-text">Before.</span></h1>
          <p className="hero-subtitle">
            Experience the future of urban mobility. Our deep learning model analyzes thousands of traffic nodes to find your fastest route in Mumbai, processing simulated environments mapped directly to Google Maps API standards.
          </p>
          <div className="hero-cta">
            <Link to="/app" className="btn btn-primary">
              Launch App <ChevronRight size={18} />
            </Link>
            <a href="#features" className="btn btn-secondary">Learn More</a>
          </div>
        </motion.div>
        
        <motion.div 
          className="hero-visual"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
        >
          <div className="glass-panel map-mockup">
            <div className="glass-header">
              <span className="dot red"></span>
              <span className="dot yellow"></span>
              <span className="dot green"></span>
            </div>
            <div className="glass-body">
              <div className="mock-route pulse-anim"></div>
              <div className="mock-marker source"></div>
              <div className="mock-marker dest"></div>
              <div className="floating-card time-card">
                 <Clock size={16} className="text-accent"/>
                 <div>
                   <span>Est. Time</span>
                   <strong>24 min</strong>
                 </div>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section id="features" className="features-section">
        <motion.div 
          className="section-header"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2>Powered by <span className="gradient-text">Deep Learning</span></h2>
          <p>Trained on high-fidelity traffic environments, our models learn to predict the unpredictable using industry-standard API structures.</p>
        </motion.div>

        <div className="features-grid">
          {features.map((f, i) => (
            <motion.div 
              key={i} 
              className="feature-card glass-panel"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              viewport={{ once: true }}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
            >
              <div className="feature-icon-wrapper">
                <f.icon className="feature-icon" />
              </div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>
      
      {/* Stats/Showcase Section */}
      <section className="stats-section">
         <motion.div 
            className="stats-container glass-panel"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
         >
            <div className="stat">
               <div className="stat-icon-wrap"><Clock size={28} className="text-accent" /></div>
               <h4>Save 20% Time</h4>
               <p>Average reduction in commute time</p>
            </div>
            <div className="stat-divider"></div>
            <div className="stat">
               <div className="stat-icon-wrap"><ShieldCheck size={28} className="text-accent" /></div>
               <h4>98% Accuracy</h4>
               <p>In traffic delay predictions</p>
            </div>
         </motion.div>
      </section>
    </div>
  );
}
