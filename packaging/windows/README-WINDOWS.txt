Capybara for Windows (amd64)
============================

Engine: llama.cpp prebuilt CPU build (see release notes for version).

Quick start
-----------
1. Extract this folder anywhere (no admin needed).
2. Install Python 3.9+ from python.org if you do not have it.
3. In a terminal, from the extracted folder:

       .\capybara.cmd pull smollm
       .\capybara.cmd run smollm "hi!"
       .\capybara.cmd serve
       .\capybara.cmd ui

Optional - add to PATH
----------------------
    setx PATH "%PATH%;<extracted folder>"

Then `capybara` works from anywhere via capybara.cmd.

Notes
-----
* Models live in %USERPROFILE%\.capybara\models
* Config:    %USERPROFILE%\.capybara\config.yaml
* OpenAI API when serving: http://127.0.0.1:11434/v1
