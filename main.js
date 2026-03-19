/* ── LOADING SCREEN ── */
(function () {
  const bar = document.getElementById('loaderBar');
  const pct = document.getElementById('loaderPct');
  const loader = document.getElementById('loader');
  let progress = 0;

  const tick = setInterval(() => {
    const inc = Math.random() * 4 + 1;
    progress = Math.min(progress + inc, 100);
    bar.style.width = progress + '%';
    pct.textContent = Math.floor(progress) + '%';
    if (progress >= 100) {
      clearInterval(tick);
      pct.textContent = '100%';
      setTimeout(() => { loader.classList.add('hidden'); }, 300);
    }
  }, 40);
})();

/* ── KST CLOCK ── */
function updateClock() {
  const now = new Date();
  const kst = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
  const h = String(kst.getHours()).padStart(2, '0');
  const m = String(kst.getMinutes()).padStart(2, '0');
  const s = String(kst.getSeconds()).padStart(2, '0');
  document.getElementById('clock').textContent = 'KST ' + h + ':' + m + ':' + s;
}
updateClock();
setInterval(updateClock, 1000);

/* ── ROTATING HERO TEXT ── */
const roles = [
  'Routine life and longevity',
  'Life hacker',
  'Startup Founder',
  'Systems Thinker'
];
let roleIdx = 0;
const rotEl = document.getElementById('rotatingText');

setInterval(() => {
  rotEl.classList.add('fade');
  setTimeout(() => {
    roleIdx = (roleIdx + 1) % roles.length;
    rotEl.textContent = roles[roleIdx];
    rotEl.classList.remove('fade');
  }, 420);
}, 2800);

/* ── MARQUEE ── */
const marqueeItems = [
  'automation', 'systems thinking', 'civic tech', 'B2G',
  'n8n', 'cross-cultural coordination', 'startup',
  '자동화', '민간안전', '디지털저항', 'longevity', 'health'
];

function buildMarquee(el) {
  marqueeItems.forEach((item, i) => {
    const span = document.createElement('span');
    span.className = 'marquee-item';
    const dot = i < marqueeItems.length - 1
      ? ' <span class="marquee-dot">·</span>' : '';
    span.innerHTML = item + dot;
    el.appendChild(span);
  });
}
buildMarquee(document.getElementById('marqueeContent'));
buildMarquee(document.getElementById('marqueeContent2'));

/* ── SCROLL REVEAL ── */
const revealEls = document.querySelectorAll('.reveal');
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      observer.unobserve(e.target);
    }
  });
}, { threshold: 0.12 });
revealEls.forEach(el => observer.observe(el));

/* ── WRITING FILTER ── */
const filterBtns = document.querySelectorAll('.filter-btn');
const writingCards = document.querySelectorAll('.writing-card');

filterBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    filterBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const filter = btn.dataset.filter;
    writingCards.forEach(card => {
      if (filter === 'all' || card.dataset.category === filter) {
        card.classList.remove('hidden');
      } else {
        card.classList.add('hidden');
      }
    });
  });
});
