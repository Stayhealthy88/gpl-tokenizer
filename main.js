/* ── LOADER ─────────────────────────────────────────── */
(function () {
  const bar    = document.getElementById('loaderBar');
  const pct    = document.getElementById('loaderPct');
  const loader = document.getElementById('loader');
  const logo   = document.querySelector('.loader-logo');

  let progress = 0;

  // Reveal logo immediately
  setTimeout(() => logo.classList.add('reveal-logo'), 200);

  const tick = setInterval(() => {
    progress += Math.random() * 3.5 + 0.5;
    if (progress > 100) progress = 100;
    bar.style.width = progress + '%';
    pct.textContent = Math.floor(progress) + '%';
    if (progress >= 100) {
      clearInterval(tick);
      pct.textContent = '100%';
      setTimeout(() => {
        loader.classList.add('hidden');
        initHero();
      }, 500);
    }
  }, 45);
})();

/* ── KST CLOCK ──────────────────────────────────────── */
function updateClock() {
  const kst = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
  const h = String(kst.getHours()).padStart(2, '0');
  const m = String(kst.getMinutes()).padStart(2, '0');
  const s = String(kst.getSeconds()).padStart(2, '0');
  const el = document.getElementById('clock');
  if (el) el.textContent = 'KST ' + h + ':' + m + ':' + s;
}
updateClock();
setInterval(updateClock, 1000);

/* ── NAVBAR SLIDE IN ────────────────────────────────── */
setTimeout(() => {
  document.querySelector('nav').classList.add('visible');
}, 1800);

/* ── HERO INIT ──────────────────────────────────────── */
function initHero() {
  // Tagline
  setTimeout(() => document.querySelector('.hero-tagline').classList.add('in'), 100);

  // Headline character split
  const headline = document.querySelector('.hero-headline');
  const text = headline.dataset.text;
  headline.innerHTML = '';
  let delay = 0;
  text.split('').forEach(ch => {
    const span = document.createElement('span');
    span.className = 'char';
    span.textContent = ch === ' ' ? '\u00A0' : ch;
    span.style.transitionDelay = delay + 's';
    headline.appendChild(span);
    delay += 0.04;
  });
  setTimeout(() => headline.classList.add('in'), 150);

  // Name stack
  const nameStack = document.querySelector('.hero-name-stack');
  if (nameStack) setTimeout(() => nameStack.classList.add('in'), 300);

  // Rest of hero
  setTimeout(() => document.querySelector('.hero-bio').classList.add('in'), 400);
  setTimeout(() => document.querySelector('.hero-ctas').classList.add('in'), 500);
  setTimeout(() => document.querySelector('.hero-stats').classList.add('in'), 600);
}

/* ── ROTATING HERO SUBTITLE ─────────────────────────── */
const roles = [
  'Routine life and longevity',
  'Life hacker',
  'Startup Founder',
  'Systems Thinker'
];
let roleIdx = 0;
const rotEl = document.getElementById('rotatingText');
if (rotEl) {
  setInterval(() => {
    rotEl.style.opacity = '0';
    rotEl.style.transform = 'translateY(8px)';
    setTimeout(() => {
      roleIdx = (roleIdx + 1) % roles.length;
      rotEl.textContent = roles[roleIdx];
      rotEl.style.opacity = '1';
      rotEl.style.transform = 'translateY(0)';
    }, 400);
  }, 3000);
}

/* ── MARQUEE DIRECTION ──────────────────────────────── */
(function () {
  const tracks = document.querySelectorAll('.marquee-track');
  let lastY = window.scrollY;
  window.addEventListener('scroll', () => {
    const dir = window.scrollY > lastY ? 1 : -1;
    lastY = window.scrollY;
    tracks.forEach(t => {
      if (dir < 0) t.classList.add('reverse');
      else t.classList.remove('reverse');
    });
  }, { passive: true });
})();

/* ── MARQUEE BUILD ──────────────────────────────────── */
(function () {
  const items = [
    'automation', 'systems thinking', 'civic tech', 'B2G', 'n8n',
    'cross-cultural', 'startup', '자동화', '민간안전', '디지털저항',
    'longevity', 'health', 'coffee'
  ];
  function build(id) {
    const el = document.getElementById(id);
    if (!el) return;
    [1, 2].forEach(() => {
      const inner = document.createElement('div');
      inner.className = 'marquee-inner';
      items.forEach((item, i) => {
        const span = document.createElement('span');
        span.className = 'marquee-item';
        span.innerHTML = item + (i < items.length - 1 ? '<span class="marquee-sep">·</span>' : '');
        inner.appendChild(span);
      });
      el.appendChild(inner);
    });
  }
  build('marqueeTrack');
})();

/* ── DRAG-SCROLL PROJECTS ───────────────────────────── */
(function () {
  const el = document.querySelector('.projects-scroll');
  if (!el) return;
  let isDown = false, startX, scrollLeft;
  el.addEventListener('mousedown', e => {
    isDown = true;
    startX = e.pageX - el.offsetLeft;
    scrollLeft = el.scrollLeft;
  });
  el.addEventListener('mouseleave', () => isDown = false);
  el.addEventListener('mouseup', () => isDown = false);
  el.addEventListener('mousemove', e => {
    if (!isDown) return;
    e.preventDefault();
    const x = e.pageX - el.offsetLeft;
    el.scrollLeft = scrollLeft - (x - startX) * 1.4;
  });
})();

/* ── SECTION HEADING LINE SPLIT ─────────────────────── */
document.querySelectorAll('.section-heading').forEach(el => {
  const lines = el.textContent.trim().split('\n').map(l => l.trim()).filter(Boolean);
  el.innerHTML = lines.map(line =>
    `<span class="line-wrap"><span class="line-inner">${line}</span></span>`
  ).join('');
});

/* ── SCROLL REVEAL (INTERSECTION OBSERVER) ──────────── */
const ioOptions = { threshold: 0.15 };

const sectionHeadings = document.querySelectorAll('.section-heading');
const io1 = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io1.unobserve(e.target); } });
}, ioOptions);
sectionHeadings.forEach(el => io1.observe(el));

const contactBig = document.querySelector('.contact-big');
if (contactBig) {
  const lines = contactBig.textContent.trim().split('\n').map(l => l.trim()).filter(Boolean);
  contactBig.innerHTML = lines.map(l =>
    `<span class="line-wrap"><span class="line-inner">${l}</span></span>`
  ).join('<br>');
  const io2 = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io2.unobserve(e.target); } });
  }, { threshold: 0.2 });
  io2.observe(contactBig);
}

// Timeline items
const timelineItems = document.querySelectorAll('.timeline-item');
const io3 = new IntersectionObserver(entries => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      e.target.style.transitionDelay = (i * 0.1) + 's';
      e.target.classList.add('in');
      io3.unobserve(e.target);
    }
  });
}, ioOptions);
timelineItems.forEach(el => io3.observe(el));

// Project cards
const projectCards = document.querySelectorAll('.project-card');
const io4 = new IntersectionObserver(entries => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      e.target.style.transitionDelay = (i * 0.12) + 's';
      e.target.classList.add('in');
      io4.unobserve(e.target);
    }
  });
}, { threshold: 0.1 });
projectCards.forEach(el => io4.observe(el));

// Writing items
const writingItems = document.querySelectorAll('.writing-item');
const io5 = new IntersectionObserver(entries => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      e.target.style.transitionDelay = (i * 0.08) + 's';
      e.target.classList.add('in');
      io5.unobserve(e.target);
    }
  });
}, ioOptions);
writingItems.forEach(el => io5.observe(el));

// Etymology items
const etymItems = document.querySelectorAll('.etymology-item, .etymology-conclusion');
const io6 = new IntersectionObserver(entries => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      e.target.style.transitionDelay = (i * 0.15) + 's';
      e.target.classList.add('in');
      io6.unobserve(e.target);
    }
  });
}, ioOptions);
etymItems.forEach(el => io6.observe(el));

/* ── WRITING FILTER ─────────────────────────────────── */
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const filter = btn.dataset.filter;
    document.querySelectorAll('.writing-item').forEach(card => {
      card.classList.toggle('hidden', filter !== 'all' && card.dataset.category !== filter);
    });
  });
});

/* ── CUSTOM CURSOR ──────────────────────────────────── */
(function () {
  const cursor = document.querySelector('.cursor');
  if (!cursor || window.innerWidth < 641) return;
  let mx = -100, my = -100;
  document.addEventListener('mousemove', e => {
    mx = e.clientX; my = e.clientY;
    cursor.style.left = mx + 'px';
    cursor.style.top  = my + 'px';
  });
  document.querySelectorAll('a, button, .project-card, .writing-item').forEach(el => {
    el.addEventListener('mouseenter', () => cursor.classList.add('expand'));
    el.addEventListener('mouseleave', () => cursor.classList.remove('expand'));
  });
})();
