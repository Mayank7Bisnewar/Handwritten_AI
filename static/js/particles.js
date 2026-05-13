/**
 * Handwritten Notes AI — particles.js
 * Animated particle canvas background
 */

(function () {
  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W, H, particles = [];
  const COUNT = 90;
  const COLORS = ['#00d4ff', '#a855f7', '#6366f1', '#f0abfc'];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  class Particle {
    constructor() { this.reset(true); }
    reset(init) {
      this.x  = Math.random() * W;
      this.y  = init ? Math.random() * H : H + 10;
      this.vx = (Math.random() - 0.5) * 0.4;
      this.vy = -(Math.random() * 0.6 + 0.2);
      this.r  = Math.random() * 1.8 + 0.4;
      this.alpha = Math.random() * 0.6 + 0.1;
      this.color = COLORS[Math.floor(Math.random() * COLORS.length)];
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.y < -10) this.reset(false);
    }
    draw() {
      ctx.save();
      ctx.globalAlpha = this.alpha;
      ctx.shadowBlur  = 8;
      ctx.shadowColor = this.color;
      ctx.fillStyle   = this.color;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  function init() {
    resize();
    particles = Array.from({ length: COUNT }, () => new Particle());
    window.addEventListener('resize', resize);
    loop();
  }

  function loop() {
    ctx.clearRect(0, 0, W, H);
    // Draw connecting lines between nearby particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.save();
          ctx.globalAlpha = (1 - dist / 120) * 0.12;
          ctx.strokeStyle = '#6366f1';
          ctx.lineWidth   = 0.8;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
          ctx.restore();
        }
      }
      particles[i].update();
      particles[i].draw();
    }
    requestAnimationFrame(loop);
  }

  window.addEventListener('DOMContentLoaded', init);
})();
