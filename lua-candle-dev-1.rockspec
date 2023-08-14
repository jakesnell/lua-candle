package = "lua-candle"
version = "dev-1"

source = {
   url = "git+https://github.com/jakesnell/lua-candle"
}

description = {
   homepage = "https://github.com/jakesnell/lua-candle",
   license = "MIT"
}

dependencies = {
   "lua >= 5.4",
   "luarocks-build-rust-mlua"
}

build = {
   type = "rust-mlua",

   install = {
      lua = {
         candle = "lua/candle.lua"
      }
   },

   modules = {
      "candle_core"
   }
}
