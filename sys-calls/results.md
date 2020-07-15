# pthread_create()
clone(child_stack=0x7fc9f1da0fb0, flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|CLONE_SETTLS|CLONE_PARENT_SETTID|CLONE_CHILD_CLEARTID, parent_tidptr=0x7fc9f1da19d0, tls=0x7fc9f1da1700, child_tidptr=0x7fc9f1da19d0) = 15992
# fork()
clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x7fc9f2f3ba10) = 15993

They both call the clone sys call. The overhead is pretty similar due to the copy-on-write

