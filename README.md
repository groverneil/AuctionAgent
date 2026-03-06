# AuctionAgent


```
        # input is of this shape:
        # return np.concatenate([
        #     items_done,
        #     current_item_onehot,
        #     [market_value_norm, current_bid_norm, my_val_current_norm],
        # ]) --> num_items + num_items + 3 = 2*num_items + 3 = input_size
```