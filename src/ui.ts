const themeToggle = document.getElementById('theme-toggle') as HTMLButtonElement;

function syncThemeToggle() {
  const dark = document.documentElement.dataset.theme !== 'light';
  themeToggle.setAttribute('aria-checked', String(dark));
  themeToggle.title = dark ? 'Switch to light theme' : 'Switch to dark theme';
}

syncThemeToggle();
themeToggle.addEventListener('click', () => {
  const theme = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  document.documentElement.dataset.theme = theme;
  syncThemeToggle();
  try {
    localStorage.setItem('waterwave-theme', theme);
  } catch { /* Theme switching still works without persistent storage. */ }
});

const panel = document.getElementById('control-content') as HTMLDivElement;
const panelToggle = document.getElementById('controls-toggle') as HTMLButtonElement;
const settings = document.getElementById('simulation-controls') as HTMLFieldSetElement;
const status = document.getElementById('loading-panel') as HTMLDivElement;
const statusTitle = status.querySelector('.loading-text') as HTMLParagraphElement;
const statusText = status.querySelector('.loading-subtext') as HTMLParagraphElement;
const reload = document.getElementById('reload-button') as HTMLButtonElement;

function setPanelOpen(open: boolean) {
  panel.hidden = !open;
  panelToggle.setAttribute('aria-expanded', String(open));
  panelToggle.setAttribute('aria-label', open ? 'Hide controls' : 'Show controls');
}

setPanelOpen(!matchMedia('(max-width: 600px)').matches);
panelToggle.addEventListener('click', () => setPanelOpen(panel.hidden === true));
document.getElementById('controls')!.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !panel.hidden) {
    setPanelOpen(false);
    panelToggle.focus();
  }
});
reload.addEventListener('click', () => location.reload());

for (const [toggleId, groupId] of [['rain-toggle', 'rain-controls'], ['caustics-toggle', 'caustics-controls']]) {
  const toggle = document.getElementById(toggleId!) as HTMLInputElement;
  const group = document.getElementById(groupId!) as HTMLFieldSetElement;
  const sync = () => { group.disabled = !toggle.checked; };
  toggle.addEventListener('input', sync);
  sync();
}

export function showLoading(title: string, description: string) {
  statusTitle.textContent = title;
  statusText.textContent = description;
  status.dataset.error = 'false';
  reload.hidden = true;
  status.classList.add('visible');
}

export function hideLoading() {
  status.classList.remove('visible');
}

export function enableSettings() {
  settings.disabled = false;
  hideLoading();
}

export function showError(title: string, description: string, fatal = false) {
  statusTitle.textContent = title;
  statusText.textContent = description;
  status.dataset.error = 'true';
  status.classList.add('visible');
  reload.hidden = false;
  if (fatal) settings.disabled = true;
}
