module {
  func.func @matmul(%a: memref<?xf64>, %b: memref<?xf64>, %c: memref<?xf64>) attributes {llvm.emit_c_interface} {
    %v = arith.constant 0 : index
    %v_2 = arith.constant 2 : index
    %v_3 = arith.constant 1 : index
    scf.for %v_4 = %v to %v_2 step %v_3 {
      %v_5 = arith.constant 4 : index
      scf.for %v_6 = %v to %v_5 step %v_3 {
        %v_7 = arith.constant 3 : index
        scf.for %v_8 = %v to %v_7 step %v_3 {
          %v_9 = arith.muli %v_4, %v_5 : index
          %v_10 = arith.addi %v_9, %v_6 : index
          %v_11 = memref.load %c[%v_10] : memref<?xf64>
          %v_12 = arith.muli %v_4, %v_7 : index
          %v_13 = arith.addi %v_12, %v_8 : index
          %v_14 = memref.load %a[%v_13] : memref<?xf64>
          %v_15 = arith.muli %v_8, %v_5 : index
          %v_16 = arith.addi %v_15, %v_6 : index
          %v_17 = memref.load %b[%v_16] : memref<?xf64>
          %v_18 = arith.mulf %v_14, %v_17 : f64
          %v_19 = arith.addf %v_11, %v_18 : f64
          %v_20 = arith.muli %v_4, %v_5 : index
          %v_21 = arith.addi %v_20, %v_6 : index
          memref.store %v_19, %c[%v_21] : memref<?xf64>
        }
      }
    }
    func.return
  }
}
