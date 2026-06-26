MLIR_HELPERS: dict[str, str] = {
    "scansearch": """\
  func.func @scansearch(%arr: memref<?xi64>, %x: i64, %lo: i64, %hi: i64) -> i64 {
    %1 = arith.constant 1 : i64
    %g:2 = scf.while (%d = %1, %p = %lo) : (i64, i64) -> (i64, i64) {
      %plt = arith.cmpi slt, %p, %hi : i64
      %cond = scf.if %plt -> (i1) {
        %pi = arith.index_cast %p : i64 to index
        %ap = memref.load %arr[%pi] : memref<?xi64>
        %al = arith.cmpi slt, %ap, %x : i64
        scf.yield %al : i1
      } else {
        %f = arith.constant false
        scf.yield %f : i1
      }
      scf.condition(%cond) %d, %p : i64, i64
    } do {
    ^bb0(%d: i64, %p: i64):
      %d2 = arith.shli %d, %1 : i64
      %p2 = arith.addi %p, %d2 : i64
      scf.yield %d2, %p2 : i64, i64
    }
    %lo1 = arith.subi %g#1, %g#0 : i64
    %minp = arith.minsi %g#1, %hi : i64
    %hi1 = arith.addi %minp, %1 : i64
    %b:2 = scf.while (%l = %lo1, %h = %hi1) : (i64, i64) -> (i64, i64) {
      %hm1 = arith.subi %h, %1 : i64
      %go = arith.cmpi slt, %l, %hm1 : i64
      scf.condition(%go) %l, %h : i64, i64
    } do {
    ^bb0(%l: i64, %h: i64):
      %diff = arith.subi %h, %l : i64
      %half = arith.shrsi %diff, %1 : i64
      %m = arith.addi %l, %half : i64
      %mi = arith.index_cast %m : i64 to index
      %am = memref.load %arr[%mi] : memref<?xi64>
      %al = arith.cmpi slt, %am, %x : i64
      %l2, %h2 = scf.if %al -> (i64, i64) {
        scf.yield %m, %h : i64, i64
      } else {
        scf.yield %l, %m : i64, i64
      }
      scf.yield %l2, %h2 : i64, i64
    }
    return %b#1 : i64
  }""",
}
