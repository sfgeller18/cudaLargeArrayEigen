**Arnoldi.hpp:**
- Fix In-Place Sorter Method for large arrays
- Template NaiveRealArnoldi so it can return complex basis matrix
- Make Krylov Iter be able to take first and lst ind (as opposed to just max_iters) to run iter starting from S-th index rather than 0.

**Shift.hpp**
- Fix the GPU restarting. Memory allocation and logic is correct but Hessenberg structure of H is breaking down after first reflection. I believe there is a truncation somewhere where information is being lost or a matrix is being overwritten.
- For now not going to modify, most restarting is to be done on 100 x 100 H matrices to bring them down to about 10 x 10