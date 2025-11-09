
```
 shared_ptr = shared_ptr:
     You can assign one shared_ptr to another.
     Both will share ownership and the reference count increases.
 
 shared_ptr = weak_ptr:
     You cannot directly assign a weak_ptr to a shared_ptr.
     You must use shared_ptr = weak_ptr.lock(), which gives a
     new shared_ptr if the object still exists, otherwise a null pointer.
 
 weak_ptr = shared_ptr:
     You can assign a shared_ptr to a weak_ptr.
     The weak_ptr will observe the object without affecting its reference count.
 
 weak_ptr = weak_ptr:
     You can assign one weak_ptr to another. Both will observe the same object.
```
