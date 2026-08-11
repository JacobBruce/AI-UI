/* Bundled into vendor.bundle.js for the isolated renderer (no Node require). */
window.$ = window.jQuery = require('jquery');
window.hljs = require('highlight.js');
window.markdownit = require('markdown-it');
window.markdownitEmoji = require('markdown-it-emoji').full;
window.markdownitFootnote = require('markdown-it-footnote');
window.markdownitTaskLists = require('markdown-it-task-lists');
window.markdownitKatex = require('@vscode/markdown-it-katex').default;
