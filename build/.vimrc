let mapleader=","

set nocompatible
set mouse-=a
set nu
set autoindent
set noexpandtab
syntax on

autocmd FileType sh,c,cpp,cc,cu,h,hpp,cc,proto,go,sh set expandtab ts=4 sw=4 textwidth=100 cindent
autocmd FileType py,java,go,cmake set expandtab ts=4 sw=4 smartindent
au BufNewFile,BufRead *.cu set ft=cuda
au BufNewFile,BufRead *.cuh set ft=cuda

"字符编码
set encoding=utf-8
let &termencoding=&encoding 
set fileencodings=utf-8,gbk 

"传说中的去掉边框用下边这一句 
set go= 
"设置配色，这里选择的是desert，也有其他方案，在vim中输入:color
"在敲tab键可以查看 
color koehler
"colorscheme monokai
"设置背景色，每种配色有两种方案，一个light、一个dark 
set background=light 
"在windows版本中vim的退格键模式默认与vi兼容，与我们的使用习惯不太符合，下边这条可以改过来 
set backspace=indent,eol,start 
"增量式搜索 
set incsearch 
"高亮搜索 
set hlsearch 
"更改注释的颜色
hi Comment ctermfg =6

"设置记录文件关闭时候的位置
if has("autocmd")
  au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
endif

"各种跳转的快捷键----
nnoremap <leader>h  <c-w>h
nnoremap <leader>j  <c-w>j
nnoremap <leader>k  <c-w>k
nnoremap <leader>l  <c-w>l

nmap <leader>bf :browse oldfiles<CR>
"nmap <leader>bc :Bclose<CR>
cnoreabbrev bc Bclose

nmap <C-h>  :bp<CR>
nmap <C-l>  :bn<CR>
"nmap <C-n>  :tabnew<CR>

" 设置ctrl+左右键切换缓冲区
nnoremap <silent> <C-left> :bnext<cr>
nnoremap <silent> <C-right> :bNext<cr>


"可视模式下的覆盖操作
"vmap <leader>r "_dP
"----

"设置折叠----
set foldmethod=manual
set foldenable              " 开始折叠
set foldmethod=syntax       " 设置语法折叠
set foldcolumn=0            " 设置折叠区域的宽度
setlocal foldlevel=1        " 设置折叠层数为
set foldlevelstart=99       " 打开文件是默认不折叠代码
"                            " 用空格键来开关折叠
nnoremap <space> @=((foldclosed(line('.')) < 0) ? 'zc' : 'zo')<CR>
"----
"set cursorline
"hi CursorLine   cterm=NONE ctermbg=NONE ctermfg=NONE guibg=NONE guifg=NONE



"come from https://github.com/VundleVim/Vundle.vim------------
set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'

" The following are examples of different formats supported.
" Keep Plugin commands between vundle#begin/end.
" plugin on GitHub repo
Plugin 'tpope/vim-fugitive'
" plugin from http://vim-scripts.org/vim/scripts.html
" Plugin 'L9'
" Git plugin not hosted on GitHub
Plugin 'git://git.wincent.com/command-t.git'
" git repos on your local machine (i.e. when working on your own plugin)
"Plugin 'file:///home/gmarik/path/to/plugin'
"
" The sparkup vim script is in a subdirectory of this repo called vim.
" Pass the path to set the runtimepath properly.
Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
" Install L9 and avoid a Naming conflict if you've already installed a
" different version somewhere else.
" Plugin 'ascenator/L9', {'name': 'newL9'}

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this line
"bundle example end----------------

"nerd tree----
Plugin 'scrooloose/nerdtree'
"set autochdir
set tags=tags;
let NERDTreeChDirMode=2
nnoremap <leader>n :NERDTreeToggle<CR>
nnoremap <leader>N :NERDTreeFind<CR>
let NERDTreeIgnore = ['\.o']
let NERDTreeShowLineNumbers=1
"let NERDTreeMapOpenInTab='<ENTER>'
"----

"nerdtree git----
"set shell=sh
"Plugin 'Xuyuanp/nerdtree-git-plugin'
"let g:NERDTreeIndicatorMapCustom = {
"    \ "Modified"  : "✹",
"    \ "Staged"    : "✚",
"    \ "Untracked" : "✭",
"    \ "Renamed"   : "➜",
"    \ "Unmerged"  : "═",
"    \ "Deleted"   : "✖",
"    \ "Dirty"     : "✗",
"    \ "Clean"     : "✔︎",
"    \ 'Ignored'   : '☒',
"    \ "Unknown"   : "?"
"    \ }
"----

"easygrep----
Plugin 'dkprice/vim-easygrep'
"----

"vim-go
Plugin 'fatih/vim-go'
let g:go_def_mode = 'godef'
"let g:go_def_mode = 'guru'
au FileType go nmap <leader>g :GoDef <C-R>=expand("<cword>")<CR><CR>
au FileType go nnoremap <leader>vg :vsp <CR>:exe "GoDef" <CR>
au FileType go nnoremap <leader>s :sp <CR>:exe "GoDef"<CR>
au FileType go nnoremap <leader>r :GoRun %<CR>
let g:go_fmt_command = "goimports"
"----

"go tag bar----
Plugin 'majutsushi/tagbar'
au FileType go nmap <F8> :TagbarToggle<CR>
"----

"vimdoc---
Plugin 'asins/vimcdoc'
set helplang=cn
"----

"python----
Plugin 'Vimjas/vim-python-pep8-indent'
autocmd FileType python setlocal foldmethod=indent
"Plugin 'Yggdroot/indentLine'
"let g:indentLine_char = '¦'
"let g:indentLine_setColors = 255
"----


"snippet----
" Track the engine.
"Plugin 'SirVer/ultisnips'
"
"" Snippets are separated from the engine. Add this if you want them:
"Plugin 'honza/vim-snippets'
"
"" Trigger configuration. Do not use <tab> if you use https://github.com/Valloric/YouCompleteMe.
"let g:UltiSnipsExpandTrigger="<tab>"
"let g:UltiSnipsJumpForwardTrigger="<c-b>"
"let g:UltiSnipsJumpBackwardTrigger="<c-z>"
"
"" If you want :UltiSnipsEdit to split your window.
"let g:UltiSnipsEditSplit="vertical"
"----

"ctags, cscope----
nnoremap <silent> <F7> :!ctags -R --c++-kinds=+p --fields=+iaS --extra=+q .<CR><CR>
nnoremap <silent> <F6> :!find . -name "*.h" -o -name "*.c" -o -name "*.cc" -o -name "*.cu" > cscope.files<CR>!cscope -Rbkq<CR><CR>
if has("cscope")
  set csto=1
  set cst
  set nocsverb
  " add any database in current directory
  if filereadable("cscope.out")
      cs add cscope.out
  endif
  set csverb
endif

set cscopetag
set nocscopeverbose
nmap <c-n>s :cs find s <C-R>=expand("<cword>")<CR><CR>	
nmap <c-n>g :cs find g <C-R>=expand("<cword>")<CR><CR>	
nmap <c-n>c :cs find c <C-R>=expand("<cword>")<CR><CR>	
nmap <c-n>t :cs find t <C-R>=expand("<cword>")<CR><CR>	
nmap <c-n>e :cs find e <C-R>=expand("<cword>")<CR><CR>	
nmap <c-n>f :cs find f <C-R>=expand("<cfile>")<CR><CR>	
nmap <c-n>i :cs find i <C-R>=expand("<cfile>")<CR>$<CR>
nmap <c-n>d :cs find d <C-R>=expand("<cword>")<CR><CR>


"跳转
au FileType cc,c,cpp,cc,cu,h,hpp nnoremap <leader>g  <c-]>
au FileType cc,c,cpp,cc,cu,h,hpp nnoremap <leader>t  <c-t>
nnoremap <leader>f  <c-f>
"----

"taglist
let Tlist_Show_One_File=1
let Tlist_Exit_OnlyWindow=1
let Tlist_Use_Right_Window=1
let Tlist_Auto_open=0
Plugin 'taglist.vim'
nnoremap <silent> <F9> :Tlist<CR><CR>
"----

"vim-airline----
"Plugin 'vim-airline/vim-airline'
"Plugin 'vim-airline/vim-airline-themes'
"let g:airline_theme='simple'
"----


"ctrp----
Plugin 'kien/ctrlp.vim'
let g:ctrlp_user_command = ['.git', 'cd %s && git ls-files -co --exclude-standard']
"----

"jedi----
Bundle 'davidhalter/jedi-vim'
au FileType py setlocal completeopt-=preview
au FileType py let g:jedi#goto_command = "<leader>G"
au FileType py let g:jedi#goto_assignments_command = "<leader>A"
au FileType py let g:jedi#goto_definitions_command = "<learder>g"
au FileType py let g:jedi#documentation_command = "<leader>K"
"au FileType py let g:jedi#usages_command = "<leader>U"
au FileType py let g:jedi#usages_command = ""
au FileType py let g:jedi#completions_command = "<C-Space>"
au FileType py let g:jedi#rename_command = "<leader>r"
"----

"Youcompleteme---
Plugin 'Valloric/YouCompleteMe'
autocmd FileType cuda set ft=c
let g:ycm_global_ycm_extra_conf = '/root/.vim/bundle/YouCompleteMe/third_party/ycmd/cpp/ycm/.ycm_extra_conf.py'
au FileType cc,c,cpp,cc,cu,h,hpp nnoremap <leader>g :YcmCompleter GoToDefinition<CR>
au FileType cc,c,cpp,cc,cu,h,hpp nnoremap <leader>t :YcmCompleter GoTo<CR>
hi YcmErrorSection guifg=#800000
"----

"minibuf---
"Bundle 'fholgado/minibufexpl.vim'
Bundle 'jlanzarotta/bufexplorer'
"let g:bufExplorerShowTabBuffer=1        " Yes.
nnoremap <silent> <Tab> :BufExplorer<CR><CR>
"----

"bufkill---
Bundle 'qpkorr/vim-bufkill'
nmap <c-x> :BD<CR><CR>
"----

"---
"Bundle 'winmanager'
"let g:winManagerWindowLayout = 'FileExplorer|BufExplorer'
"----
