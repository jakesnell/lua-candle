(local candle (require "candle"))

;; (print (candle.ones [1 2]))
;; (print (candle.ones [1]))
;; (print (candle.zeros [3 2 1]))

(local a (candle.tensor 9))
(local b (candle.tensor 3))
(print (+ a b))
(print (- a b))
(print (* a b))

(local c (candle.tensor 0))
(print (/ a c))

(local d (candle.rand [3 5]))
(local e (candle.randn [3 5]))
(print d)
(print e)
(print (+ d e))

(local f (: (candle.randn [200 3]) :sum_all))
(print f)

(local g (candle.ones [2 3] "f16"))
(print g)

(local h (candle.ones [2 1] "u8"))
(print h)
