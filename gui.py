#!/usr/bin/env python3
import json, os, subprocess, sys, threading, urllib.error, urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

HOME=Path(os.environ.get('CAPYBARA_HOME',str(Path.home()/'.capybara')))
MODELS=Path(os.environ.get('CAPYBARA_MODELS',str(HOME/'models')))
CLI=HOME/'bin'/'capybara.py'; PORT=int(os.environ.get('CAPYBARA_PORT','11434'))
BASE=f'http://127.0.0.1:{PORT}'

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('Capybara'); self.geometry('1080x720'); self.minsize(900,600); self.configure(bg='#111318')
        self.model=tk.StringVar(); self.status=tk.StringVar(value='Stopped'); self.context=tk.StringVar(value=os.environ.get('CAPYBARA_CONTEXT','10240'))
        self._build(); self.refresh_models(); self.after(1500,self.poll)

    def _build(self):
        style=ttk.Style(self); style.theme_use('clam'); style.configure('TFrame',background='#111318'); style.configure('TLabel',background='#111318',foreground='#e7e9ee'); style.configure('TButton',padding=8); style.configure('TCombobox',padding=6)
        top=ttk.Frame(self,padding=16); top.pack(fill='x')
        ttk.Label(top,text='Capybara',font=('TkDefaultFont',22,'bold')).pack(side='left')
        ttk.Label(top,text='local LLM engine',font=('TkDefaultFont',11)).pack(side='left',padx=12)
        ttk.Label(top,textvariable=self.status).pack(side='right')
        bar=ttk.Frame(self,padding=(16,0,16,10)); bar.pack(fill='x')
        ttk.Label(bar,text='Model').pack(side='left')
        self.combo=ttk.Combobox(bar,textvariable=self.model,state='readonly',width=42); self.combo.pack(side='left',padx=8)
        ttk.Button(bar,text='Refresh',command=self.refresh_models).pack(side='left')
        ttk.Button(bar,text='Import GGUF',command=self.import_model).pack(side='left',padx=6)
        ttk.Button(bar,text='Pull HF',command=self.pull_hf).pack(side='left')
        ttk.Button(bar,text='Start',command=self.start).pack(side='right')
        ttk.Button(bar,text='Stop',command=self.stop).pack(side='right',padx=6)

        pane=tk.PanedWindow(self,orient='horizontal',bg='#111318',sashwidth=4,bd=0,highlightthickness=0); pane.pack(fill='both',expand=True,padx=16,pady=(0,16))
        left=ttk.Frame(pane,padding=10); right=ttk.Frame(pane,padding=10); pane.add(left,minsize=240); pane.add(right,minsize=550)
        ttk.Label(left,text='Installed models',font=('TkDefaultFont',13,'bold')).pack(anchor='w')
        self.listbox=tk.Listbox(left,bg='#181b22',fg='#e7e9ee',selectbackground='#2c5cff',borderwidth=0,highlightthickness=0); self.listbox.pack(fill='both',expand=True,pady=8); self.listbox.bind('<<ListboxSelect>>',self.select_model)
        ttk.Button(left,text='Delete',command=self.delete_model).pack(fill='x')
        ttk.Button(left,text='Show info',command=self.show_info).pack(fill='x',pady=(6,0))

        self.chat=tk.Text(right,bg='#181b22',fg='#e7e9ee',insertbackground='white',wrap='word',borderwidth=0,padx=12,pady=12); self.chat.pack(fill='both',expand=True)
        bottom=ttk.Frame(right); bottom.pack(fill='x',pady=(8,0));
        self.entry=tk.Entry(bottom,bg='#181b22',fg='#e7e9ee',insertbackground='white',relief='flat'); self.entry.pack(side='left',fill='x',expand=True,ipady=8); self.entry.bind('<Return>',lambda e:self.send())
        ttk.Button(bottom,text='Send',command=self.send).pack(side='left',padx=(8,0))
        opt=ttk.Frame(right); opt.pack(fill='x',pady=(8,0)); ttk.Label(opt,text='Context').pack(side='left'); ttk.Entry(opt,textvariable=self.context,width=10).pack(side='left',padx=6); ttk.Label(opt,text='10K default').pack(side='left')

    def models(self): return sorted(MODELS.glob('*.gguf'))
    def refresh_models(self):
        ms=self.models(); names=[p.name for p in ms]; self.listbox.delete(0,'end'); [self.listbox.insert('end',n) for n in names]; self.combo['values']=names
        if names and not self.model.get(): self.model.set(names[0])
    def select_model(self,_=None):
        s=self.listbox.curselection();
        if s: self.model.set(self.listbox.get(s[0]))
    def selected(self):
        n=self.model.get(); return MODELS/n if n else None
    def run_cli(self,*args): return subprocess.run([sys.executable,str(CLI),*args],capture_output=True,text=True)
    def import_model(self):
        f=filedialog.askopenfilename(filetypes=[('GGUF model','*.gguf')]);
        if not f:return
        r=self.run_cli('pull',f)
        if r.returncode: messagebox.showerror('Capybara',r.stderr)
        self.refresh_models()
    def pull_hf(self):
        win=tk.Toplevel(self); win.title('Pull from Hugging Face'); win.geometry('500x160'); win.transient(self); win.grab_set()
        ttk.Label(win,text='HF repo or repo:quant').pack(padx=16,pady=(18,4),anchor='w'); e=ttk.Entry(win); e.pack(fill='x',padx=16); e.insert(0,'tensorblock/SmolLM2-135M-Instruct-GGUF:Q2_K')
        def go():
            spec=e.get().strip(); win.destroy(); self.status.set('Downloading...')
            threading.Thread(target=self._pull,args=(spec,),daemon=True).start()
        ttk.Button(win,text='Download',command=go).pack(pady=12)
    def _pull(self,spec):
        r=self.run_cli('pull',spec); self.after(0,lambda:self._done(r,'Pull'))
    def _done(self,r,title):
        self.status.set('Ready' if r.returncode==0 else 'Error'); self.refresh_models();
        if r.returncode: messagebox.showerror(title,r.stderr)
    def start(self):
        p=self.selected();
        if not p: messagebox.showwarning('Capybara','Select a model'); return
        self.status.set('Starting...')
        env=os.environ.copy(); env['CAPYBARA_CONTEXT']=self.context.get(); env['CAPYBARA_MODEL']=str(p)
        subprocess.Popen([sys.executable,str(CLI),'serve'],env=env)
    def stop(self): self.run_cli('stop'); self.status.set('Stopped')
    def poll(self):
        try:
            urllib.request.urlopen(BASE+'/health',timeout=1); self.status.set('Running')
        except Exception: self.status.set('Stopped')
        self.after(1500,self.poll)
    def send(self):
        text=self.entry.get().strip();
        if not text:return
        p=self.selected();
        if not p: messagebox.showwarning('Capybara','Select a model'); return
        self.entry.delete(0,'end'); self.chat.insert('end',f'You: {text}\n\n'); self.chat.see('end'); self.status.set('Thinking...')
        threading.Thread(target=self._chat,args=(p,text),daemon=True).start()
    def _chat(self,prompt_model,text):
        try:
            try: urllib.request.urlopen(BASE+'/health',timeout=1)
            except Exception:
                env=os.environ.copy(); env['CAPYBARA_MODEL']=str(prompt_model); env['CAPYBARA_CONTEXT']=self.context.get(); subprocess.Popen([sys.executable,str(CLI),'serve'],env=env)
                import time; time.sleep(2)
            data=json.dumps({'model':prompt_model.name,'messages':[{'role':'user','content':text}],'stream':False}).encode(); req=urllib.request.Request(BASE+'/v1/chat/completions',data=data,headers={'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=None) as r: obj=json.loads(r.read())
            out=obj['choices'][0]['message']['content']
            self.after(0,lambda:self._append(out))
        except Exception as e: self.after(0,lambda:self._append(f'[error] {e}'))
    def _append(self,text): self.chat.insert('end',f'Capybara: {text}\n\n'); self.chat.see('end'); self.status.set('Running')
    def delete_model(self):
        p=self.selected();
        if not p:return
        if messagebox.askyesno('Delete',f'Delete {p.name}?'): self.run_cli('rm',p.name); self.refresh_models()
    def show_info(self):
        p=self.selected();
        if p: messagebox.showinfo('Model info',f'{p.name}\n\n{p.stat().st_size:,} bytes\n\n{p}')

if __name__=='__main__': App().mainloop()
