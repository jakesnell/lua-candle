local candle = require "candle"

describe("candle", function()
  describe("should have tensors", function()
    it("should implement zeros", function()
      local tensor = candle.zeros({ 5, 2 }, candle.f32)
      local shape = tensor:shape()
      assert.are.same(shape, { 5, 2 })
      local rank = tensor:rank()
      assert.are.equal(rank, 2)
    end)

    it("should implement initialization from a table", function()
      local tensor = candle.Tensor({ 3., 1., 4. })
      local shape = tensor:shape()
      assert.are.same(shape, { 3 })
    end)
  end)
end)
