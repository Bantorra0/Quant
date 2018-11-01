import unittest
import trading


class TraderTestCase(unittest.TestCase):
    def test_order_buy(self):
        trader = trading.Trader()
        amt = 100000
        account = trading.Account(init_amt=100000)

        orders = [("600345",300,24.5),("002345",500,11.2),("600229",1200,9.62),("600229",100,9.32)]
        for code,cnt,price in orders:
            trader.order_buy(code,cnt,price,account)
            amt -= cnt * price
            self.assertEqual(amt, account.cash)

        pos = {"600345":{24.5:300},"002345":{11.2:500},"600229":{9.62:1200,9.32:100}}
        self.assertEqual(pos, account.stocks)

    def test_order_sell(self):
        trader = trading.Trader()
        amt = 100000
        account = trading.Account(init_amt=100000)

        orders_buy = [("600345", 300, 24.5), ("002345", 500, 11.2), ("600229", 1200, 9.62), ("600229", 100, 9.32)]
        for code, cnt, price in orders_buy:
            trader.order_buy(code, cnt, price,account)

        # Include a case that buys 600345 once and sells all, the code should be del from stocks's key.
        # Include a case that buys and sells the same amount with the same price,
        #       the price should be del from stocks[code]'s key.
        orders_sell = [("600345", 300, 26.5), ("002345", 300, 12.2),("002345", 100, 12.5), ("600229", 600, 9.72), ("600229", 100, 9.32)]
        for code, cnt, price in orders_sell:
            trader.order_sell(code,cnt,price,account)
        pos = {"002345":{11.2:500,12.2:-300,12.5:-100},"600229":{9.62:1200,9.72:-600}}
        self.assertEqual(pos,account.stocks)

        # Include a case that total amount of buy and sell is 0, the code should be del from stocks.
        orders_sell = [("600229", 600, 9.52)]
        for code, cnt, price in orders_sell:
            trader.order_sell(code,cnt,price,account)
        pos = {"002345":{11.2:500,12.2:-300,12.5:-100}}
        self.assertEqual(pos,account.stocks)


if __name__ == '__main__':
    unittest.main()
