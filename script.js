const canvas = document.getElementById('constellation');
const ctx = canvas.getContext('2d');
let particles = [];
let mouse = { x: null, y: null, radius: 140 };

const PARTICLE_COUNT = 110;

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

function createParticles() {
  particles = [];
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      size: Math.random() * 2 + 0.6
    });
  }
}

function drawParticles() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = 'rgba(92, 242, 255, 0.6)';
  particles.forEach((particle) => {
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
    ctx.fill();
  });
}

function connectParticles() {
  for (let a = 0; a < particles.length; a++) {
    for (let b = a + 1; b < particles.length; b++) {
      const dx = particles[a].x - particles[b].x;
      const dy = particles[a].y - particles[b].y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < 120) {
        const opacity = 1 - distance / 120;
        ctx.strokeStyle = `rgba(92, 242, 255, ${opacity * 0.35})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(particles[a].x, particles[a].y);
        ctx.lineTo(particles[b].x, particles[b].y);
        ctx.stroke();
      }
    }
  }
}

function updateParticles() {
  particles.forEach((particle) => {
    particle.x += particle.vx;
    particle.y += particle.vy;

    if (particle.x < 0 || particle.x > canvas.width) {
      particle.vx *= -1;
    }
    if (particle.y < 0 || particle.y > canvas.height) {
      particle.vy *= -1;
    }

    if (mouse.x && mouse.y) {
      const dx = particle.x - mouse.x;
      const dy = particle.y - mouse.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < mouse.radius) {
        particle.x += dx * 0.02;
        particle.y += dy * 0.02;
      }
    }
  });
}

function animate() {
  updateParticles();
  drawParticles();
  connectParticles();
  requestAnimationFrame(animate);
}

function initConstellation() {
  resizeCanvas();
  createParticles();
  animate();
}

window.addEventListener('resize', () => {
  resizeCanvas();
  createParticles();
});

window.addEventListener('mousemove', (event) => {
  mouse.x = event.clientX;
  mouse.y = event.clientY;
});

window.addEventListener('mouseout', () => {
  mouse.x = null;
  mouse.y = null;
});

initConstellation();

// Reveal animations
const revealElements = document.querySelectorAll('.card, .timeline-item, .panel-glass, .module-card, .use-case-grid article, .metric-card, .cta');
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  },
  {
    threshold: 0.2
  }
);

revealElements.forEach((el) => {
  el.classList.add('reveal');
  observer.observe(el);
});

// Module selector interaction
const selectorButtons = document.querySelectorAll('.selector-button');
const modulePanes = document.querySelectorAll('.module-pane');

selectorButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const target = button.dataset.module;

    selectorButtons.forEach((btn) => btn.classList.toggle('active', btn === button));
    modulePanes.forEach((pane) => {
      pane.classList.toggle('active', pane.dataset.module === target);
    });
  });
});

// Floating effect for hero metrics
const metrics = document.querySelectorAll('.metric-card, .module-card');
let floatingTick = 0;

function floatCards() {
  floatingTick += 0.01;
  metrics.forEach((card, index) => {
    const offset = Math.sin(floatingTick + index) * 2.5;
    card.style.transform = `translateY(${offset}px)`;
  });
  requestAnimationFrame(floatCards);
}

floatCards();
