(local candle (require "candle"))

;; (print (candle.ones [1 2]))
;; (print (candle.ones [1]))
;; (print (candle.zeros [3 2 1]))

(local a (candle.Tensor 9))
(local b (candle.Tensor 3))
(print (+ a b))
(print (- a b))
(print (* a b))

(local c (candle.Tensor 0))
(print (/ a c))

(local d (candle.rand [3 5]))
(local e (candle.randn [3 5]))
(print d)
(print e)
(print (+ d e))

(local f (candle.randn [3 3]))
(print f)
