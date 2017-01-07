# -*- coding: utf-8 -*-  

import os
import redis

def main():
    r = redis.Redis(host='localhost', port=6379, db=0)   #如果设置了密码，就加上password=密码
    r.set('foo', 'bar')   #或者写成 r['foo'] = 'bar'
    ret = r.get('foo')
    print ret

    # 新建一条键名为"123456"的数据, 包含属性attr_1  
    r.hset("123456", "attr_1", 100)  
    # 更改键名为"123456"的数据, 更改属性attr_1的值  
    r.hset("123456", "attr_1", 200)  
  
    # 取出属性attr_1的值  
    attr_1 = r.hget("123456", "attr_1")  
  
    # 输出看一下(发现属性值已经为str)  
    print "-- get attr_1:", attr_1  

if __name__ == '__main__':
    main()
