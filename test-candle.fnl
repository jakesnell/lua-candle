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

;; (local f (: (candle.randn [200 3] :f64) :sum_all))
;; (print (/ f (candle.tensor 600.)))

;; (local g (candle.ones [2 3] "f16"))
;; (print g)

(local h (candle.ones [2 1] "u8"))
(print h)
(print (h:to :f16))
(print (h:to candle.f64))
(print (h:to :bf16))

(local f (candle.randn [3]))
(print f)

(local p (candle.randn [2 3]))
(local q (candle.randn [3 4]))
(print p)
(print q)
(print (p:matmul q))
