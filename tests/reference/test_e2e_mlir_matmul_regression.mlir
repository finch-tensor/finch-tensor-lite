module {
  func.func @main(%_arg1: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_arg2: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_arg3: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_arg1[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_arg1[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_4 = arith.index_cast %v_3 : i64 to index
    %v_5 = llvm.extractvalue %_arg1[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_6 = arith.index_cast %v_5 : i64 to index
    %v_7 = llvm.extractvalue %_arg2[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_8 = builtin.unrealized_conversion_cast %v_7 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_9 = llvm.extractvalue %_arg2[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_arg2[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_12 = arith.index_cast %v_11 : i64 to index
    %v_13 = llvm.extractvalue %_arg3[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_15 = llvm.extractvalue %_arg3[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_16 = arith.index_cast %v_15 : i64 to index
    %v_17 = llvm.extractvalue %_arg3[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_18 = arith.index_cast %v_17 : i64 to index
    %v_19 = arith.constant 0 : index
    %v_20 = memref.dim %v_14, %v_19 : memref<?xf64>
    %v_21 = arith.constant 1 : index
    scf.for %v_22 = %v_19 to %v_20 step %v_21 {
      %v_23 = arith.constant 0.0 : f64
      memref.store %v_23, %v_14[%v_22] : memref<?xf64>
    }
    scf.for %v_24 = %v_19 to %v_4 step %v_21 {
      %v_25 = llvm.extractvalue %_arg3[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_26 = arith.index_cast %v_25 : i64 to index
      %v_27 = arith.muli %v_26, %v_24 : index
      %v_28 = arith.addi %v_19, %v_27 : index
      %v_29 = llvm.extractvalue %_arg1[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_30 = arith.index_cast %v_29 : i64 to index
      %v_31 = arith.muli %v_30, %v_24 : index
      %v_32 = arith.addi %v_19, %v_31 : index
      scf.for %v_33 = %v_19 to %v_6 step %v_21 {
        %v_34 = llvm.extractvalue %_arg1[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
        %v_35 = arith.index_cast %v_34 : i64 to index
        %v_36 = arith.muli %v_35, %v_33 : index
        %v_37 = arith.addi %v_32, %v_36 : index
        %v_38 = llvm.extractvalue %_arg2[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
        %v_39 = arith.index_cast %v_38 : i64 to index
        %v_40 = arith.muli %v_39, %v_33 : index
        %v_41 = arith.addi %v_19, %v_40 : index
        scf.for %v_42 = %v_19 to %v_12 step %v_21 {
          %v_43 = llvm.extractvalue %_arg3[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_44 = arith.index_cast %v_43 : i64 to index
          %v_45 = arith.muli %v_44, %v_42 : index
          %v_46 = arith.addi %v_28, %v_45 : index
          %v_47 = llvm.extractvalue %_arg2[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
          %v_48 = arith.index_cast %v_47 : i64 to index
          %v_49 = arith.muli %v_48, %v_42 : index
          %v_50 = arith.addi %v_41, %v_49 : index
          %v_51 = memref.load %v_14[%v_46] : memref<?xf64>
          %v_52 = memref.load %v_2[%v_37] : memref<?xf64>
          %v_53 = memref.load %v_8[%v_50] : memref<?xf64>
          %v_54 = arith.mulf %v_52, %v_53 : f64
          %v_55 = arith.addf %v_51, %v_54 : f64
          memref.store %v_55, %v_14[%v_46] : memref<?xf64>
        }
      }
    }
    %v_56 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_57 = llvm.insertvalue %_arg3, %v_56[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    func.return
  }
}
