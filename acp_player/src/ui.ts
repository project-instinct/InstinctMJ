// 轻量自定义下拉：两色样式、可搜索、支持键盘（↑↓ 选择，Enter 确认，Esc 关闭）。
// 的 API 只保留这个页面需要的那一点：options / value / disabled / onChange。

export interface SelectItem {
  value: string;
  label: string;
  /** extra text the dropdown search should match without showing it in the label */
  keywords?: string;
}

export interface SelectOptions {
  search?: boolean;
  placeholder?: string;
  emptyText?: string;
}

type ChangeHandler = (value: string) => void;

function element<T extends HTMLElement>(tag: string, className?: string): T {
  const node = document.createElement(tag) as T;
  if (className) node.className = className;
  return node;
}

export class Select {
  static instances = new Set<Select>();

  root: HTMLElement;
  searchable: boolean;
  placeholder: string;
  emptyText: string;
  items: SelectItem[] = [];
  disabled = false;
  open = false;
  active = -1;
  onChange: ChangeHandler | null = null;
  _value = '';
  button: HTMLButtonElement;
  caption: HTMLElement;
  caret: HTMLElement;
  menu: HTMLElement;
  searchInput: HTMLInputElement | null = null;
  list: HTMLElement;
  rows: HTMLButtonElement[] = [];

  constructor(root: HTMLElement, { search = false, placeholder = '请选择', emptyText = '没有匹配项' }: SelectOptions = {}) {
    this.root = root;
    this.searchable = search;
    this.placeholder = placeholder;
    this.emptyText = emptyText;

    this.root.classList.add('sel');
    this.root.innerHTML = '';

    this.button = element<HTMLButtonElement>('button', 'sel-btn');
    this.button.type = 'button';
    this.caption = element('span', 'sel-label');
    this.caret = element('span', 'sel-caret');
    this.caret.textContent = '▾';
    this.button.append(this.caption, this.caret);

    this.menu = element('div', 'sel-menu');
    this.menu.hidden = true;
    if (this.searchable) {
      this.searchInput = element<HTMLInputElement>('input', 'sel-search');
      this.searchInput.type = 'text';
      this.searchInput.placeholder = '输入关键字过滤…';
      this.searchInput.autocomplete = 'off';
      this.menu.appendChild(this.searchInput);
    }
    this.list = element('div', 'sel-list');
    this.list.setAttribute('role', 'listbox');
    this.menu.appendChild(this.list);
    this.root.append(this.button, this.menu);

    this.button.addEventListener('click', (event) => {
      event.stopPropagation();
      this.toggle();
    });
    this.button.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowDown' || event.key === 'Enter') {
        event.preventDefault();
        this.show();
      }
    });
    this.list.addEventListener('keydown', (event) => this.onKey(event));
    if (this.searchInput) this.searchInput.addEventListener('keydown', (event) => this.onKey(event));
    if (this.searchInput) this.searchInput.addEventListener('input', () => {
      this.active = -1;
      this.renderList();
    });
    document.addEventListener('pointerdown', (event) => {
      if (this.open && !this.root.contains(event.target as Node)) this.close();
    });
    window.addEventListener('resize', () => this.close());
    Select.instances.add(this);
    this.refresh();
  }

  get value(): string { return this._value; }

  set value(next: string | number | null | undefined) {
    this._value = next == null ? '' : String(next);
    this.refresh();
  }

  setOptions(items: Array<SelectItem | string>): void {
    this.items = items.map((item) => (typeof item === 'string' ? { value: item, label: item } : item));
    if (this._value && !this.items.some((item) => item.value === this._value)) this._value = '';
    this.renderList();
    this.refresh();
  }

  get selected(): SelectItem | null {
    return this.items.find((item) => item.value === this._value) || null;
  }

  refresh(): void {
    this.root.classList.toggle('is-disabled', this.disabled);
    this.button.disabled = this.disabled;
    const match = this.selected;
    this.caption.textContent = match ? match.label
      : this.disabled ? this.emptyText
        : this._value || this.placeholder;
    this.caption.classList.toggle('is-empty', !match);
    this.button.title = match ? match.label : '';
  }

  toggle(): void { this.open ? this.close() : this.show(); }

  show(): void {
    if (this.disabled || this.open) return;
    this.closeOthers();
    this.open = true;
    this.menu.hidden = false;
    this.root.classList.add('open');
    if (this.searchInput) {
      this.searchInput.value = '';
      this.searchInput.focus();
      this.searchInput.select();
    } else {
      this.list.focus();
    }
    this.active = this.visibleItems().findIndex((item) => item.value === this._value);
    this.renderList();
    const row = this.active >= 0 ? this.rows[this.active] : null;
    if (row) row.scrollIntoView({ block: 'nearest' });
  }

  close(): void {
    this.open = false;
    this.menu.hidden = true;
    this.root.classList.remove('open');
  }

  closeOthers(): void {
    for (const other of Select.instances) if (other !== this) other.close();
  }

  visibleItems(): SelectItem[] {
    const query = this.searchInput ? this.searchInput.value.trim().toLowerCase() : '';
    if (!query) return this.items;
    return this.items.filter((item) => `${item.label} ${item.value} ${item.keywords || ''}`.toLowerCase().includes(query));
  }

  renderList(): void {
    this.list.textContent = '';
    this.rows = [];
    const items = this.visibleItems();
    if (!items.length) {
      const empty = element<HTMLElement>('div', 'sel-empty');
      empty.textContent = this.emptyText;
      this.list.appendChild(empty);
      return;
    }
    items.forEach((item, index) => {
      const row = element<HTMLButtonElement>('button', 'sel-item');
      row.type = 'button';
      row.setAttribute('role', 'option');
      row.setAttribute('aria-selected', String(item.value === this._value));
      row.classList.toggle('active', index === this.active);
      row.textContent = item.label;
      row.title = item.value;
      row.addEventListener('click', (event) => {
        event.stopPropagation();
        this.pick(item.value);
      });
      row.addEventListener('pointermove', () => {
        if (this.active === index) return;
        this.active = index;
        this.paintActive();
      });
      this.rows.push(row);
      this.list.appendChild(row);
    });
    if (this.active >= items.length) this.active = items.length - 1;
    this.paintActive();
  }

  paintActive(): void {
    this.rows.forEach((row, index) => row.classList.toggle('active', index === this.active));
  }

  pick(value: string): void {
    const changed = value !== this._value;
    this._value = value;
    this.close();
    this.refresh();
    if (changed && this.onChange) this.onChange(value);
    this.button.focus();
  }

  onKey(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      event.preventDefault();
      this.close();
      this.button.focus();
      return;
    }
    if (event.key === 'Tab') { this.close(); return; }
    const items = this.visibleItems();
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (!items.length) return;
      const step = event.key === 'ArrowDown' ? 1 : -1;
      this.active = (this.active + step + items.length) % items.length;
      this.paintActive();
      const row = this.rows[this.active];
      if (row) row.scrollIntoView({ block: 'nearest' });
    } else if (event.key === 'Enter') {
      event.preventDefault();
      const item = items[this.active < 0 ? 0 : this.active];
      if (item) this.pick(item.value);
    }
  }
}
