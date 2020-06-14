ID_COLUMNS = [
    'constant_id',
    'id',
    'item_id',
    'dept_id',
    'cat_id',
    'store_id',
    'state_id'
]

LEVELS = [
    'constant_id',  # for sum by all items
    'state_id',
    'store_id',
    'cat_id',
    'dept_id',
    ['state_id', 'cat_id'],
    ['state_id', 'dept_id'],
    ['store_id', 'cat_id'],
    ['store_id', 'dept_id'],
    'item_id',
    ['state_id', 'item_id'],
    ['store_id', 'item_id'],
]
