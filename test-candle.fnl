(local candle (require "candle"))

;; (print (candle.ones [1 2]))
;; (print (candle.ones [1]))
;; (print (candle.zeros [3 2 1]))

(local a (candle.Tensor 9))
(local b (candle.Tensor 3))
(print (+ a b))
(print (- a b))
(print (- a b b))
