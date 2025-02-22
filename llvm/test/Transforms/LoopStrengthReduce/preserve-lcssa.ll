; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 2
; RUN: opt -S -loop-reduce %s | FileCheck --check-prefixes=LEGACYPM %s
; RUN: opt -S -passes=loop-reduce %s | FileCheck --check-prefixes=NEWPM %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @function_0(i32 %val_i32_8, i32 %val_i32_9) {
; LEGACYPM-LABEL: define void @function_0
; LEGACYPM-SAME: (i32 [[VAL_I32_8:%.*]], i32 [[VAL_I32_9:%.*]]) {
; LEGACYPM-NEXT:    [[VAL_I1_22:%.*]] = trunc i8 -66 to i1
; LEGACYPM-NEXT:    br i1 [[VAL_I1_22]], label [[BB_2_PREHEADER:%.*]], label [[BB_2_PREHEADER]]
; LEGACYPM:       bb_2.preheader:
; LEGACYPM-NEXT:    br label [[BB_2:%.*]]
; LEGACYPM:       bb_2:
; LEGACYPM-NEXT:    br label [[PRHDR_LOOP_3:%.*]]
; LEGACYPM:       prhdr_loop_3:
; LEGACYPM-NEXT:    br label [[LOOP_4:%.*]]
; LEGACYPM:       loop_4:
; LEGACYPM-NEXT:    [[LSR_IV:%.*]] = phi i32 [ [[LSR_IV_NEXT:%.*]], [[BE_6:%.*]] ], [ 7851, [[PRHDR_LOOP_3]] ]
; LEGACYPM-NEXT:    br i1 [[VAL_I1_22]], label [[BE_6]], label [[LOOP_EXIT_7SPLIT:%.*]]
; LEGACYPM:       bb_5:
; LEGACYPM-NEXT:    [[VAL_I32_40:%.*]] = mul i32 [[VAL_I32_9]], [[VAL_I32_24_LCSSA:%.*]]
; LEGACYPM-NEXT:    br label [[BB_2]]
; LEGACYPM:       be_6:
; LEGACYPM-NEXT:    [[LSR_IV_NEXT]] = add i32 [[LSR_IV]], 1
; LEGACYPM-NEXT:    br i1 [[VAL_I1_22]], label [[LOOP_4]], label [[BE_6_LOOP_EXIT_7_CRIT_EDGE:%.*]]
; LEGACYPM:       loop_exit_7split:
; LEGACYPM-NEXT:    [[LSR_IV_LCSSA:%.*]] = phi i32 [ [[LSR_IV]], [[LOOP_4]] ]
; LEGACYPM-NEXT:    br label [[LOOP_EXIT_7:%.*]]
; LEGACYPM:       be_6.loop_exit_7_crit_edge:
; LEGACYPM-NEXT:    [[LSR_IV_LCSSA1:%.*]] = phi i32 [ [[LSR_IV]], [[BE_6]] ]
; LEGACYPM-NEXT:    br label [[LOOP_EXIT_7]]
; LEGACYPM:       loop_exit_7:
; LEGACYPM-NEXT:    [[VAL_I32_24_LCSSA]] = phi i32 [ [[LSR_IV_LCSSA1]], [[BE_6_LOOP_EXIT_7_CRIT_EDGE]] ], [ [[LSR_IV_LCSSA]], [[LOOP_EXIT_7SPLIT]] ]
; LEGACYPM-NEXT:    br label [[BB_5:%.*]]
;
; NEWPM-LABEL: define void @function_0
; NEWPM-SAME: (i32 [[VAL_I32_8:%.*]], i32 [[VAL_I32_9:%.*]]) {
; NEWPM-NEXT:    [[VAL_I1_22:%.*]] = trunc i8 -66 to i1
; NEWPM-NEXT:    br i1 [[VAL_I1_22]], label [[BB_2_PREHEADER:%.*]], label [[BB_2_PREHEADER]]
; NEWPM:       bb_2.preheader:
; NEWPM-NEXT:    br label [[BB_2:%.*]]
; NEWPM:       bb_2:
; NEWPM-NEXT:    br label [[PRHDR_LOOP_3:%.*]]
; NEWPM:       prhdr_loop_3:
; NEWPM-NEXT:    br label [[LOOP_4:%.*]]
; NEWPM:       loop_4:
; NEWPM-NEXT:    [[LSR_IV:%.*]] = phi i32 [ [[LSR_IV_NEXT:%.*]], [[BE_6:%.*]] ], [ 7851, [[PRHDR_LOOP_3]] ]
; NEWPM-NEXT:    br i1 [[VAL_I1_22]], label [[BE_6]], label [[LOOP_EXIT_7SPLIT:%.*]]
; NEWPM:       bb_5:
; NEWPM-NEXT:    [[VAL_I32_40:%.*]] = mul i32 [[VAL_I32_9]], [[VAL_I32_24_LCSSA:%.*]]
; NEWPM-NEXT:    br label [[BB_2]]
; NEWPM:       be_6:
; NEWPM-NEXT:    [[LSR_IV_NEXT]] = add i32 [[LSR_IV]], 1
; NEWPM-NEXT:    br i1 [[VAL_I1_22]], label [[LOOP_4]], label [[BE_6_LOOP_EXIT_7_CRIT_EDGE:%.*]]
; NEWPM:       loop_exit_7split:
; NEWPM-NEXT:    [[LSR_IV_LCSSA:%.*]] = phi i32 [ [[LSR_IV]], [[LOOP_4]] ]
; NEWPM-NEXT:    br label [[LOOP_EXIT_7:%.*]]
; NEWPM:       be_6.loop_exit_7_crit_edge:
; NEWPM-NEXT:    [[LSR_IV_LCSSA1:%.*]] = phi i32 [ [[LSR_IV]], [[BE_6]] ]
; NEWPM-NEXT:    br label [[LOOP_EXIT_7]]
; NEWPM:       loop_exit_7:
; NEWPM-NEXT:    [[VAL_I32_24_LCSSA]] = phi i32 [ [[LSR_IV_LCSSA1]], [[BE_6_LOOP_EXIT_7_CRIT_EDGE]] ], [ [[LSR_IV_LCSSA]], [[LOOP_EXIT_7SPLIT]] ]
; NEWPM-NEXT:    br label [[BB_5:%.*]]
;
  %val_i1_22 = trunc i8 -66 to i1
  br i1 %val_i1_22, label %bb_2, label %bb_2

bb_2:                                             ; preds = %bb_5, %entry_1, %entry_1
  br label %prhdr_loop_3

prhdr_loop_3:                                     ; preds = %bb_2
  br label %loop_4

loop_4:                                           ; preds = %be_6, %prhdr_loop_3
  %loop_cnt_i32_11 = phi i32 [ 7850, %prhdr_loop_3 ], [ %val_i32_24, %be_6 ]
  %val_i32_24 = add i32 %loop_cnt_i32_11, 1
  br i1 %val_i1_22, label %be_6, label %loop_exit_7

bb_5:                                             ; preds = %loop_exit_7
  %val_i32_40 = mul i32 %val_i32_9, %val_i32_24.lcssa
  br label %bb_2

be_6:                                             ; preds = %loop_4
  br i1 %val_i1_22, label %loop_4, label %loop_exit_7

loop_exit_7:                                      ; preds = %be_6, %loop_4
  %val_i32_24.lcssa = phi i32 [ %val_i32_24, %be_6 ], [ %val_i32_24, %loop_4 ]
  br label %bb_5
}

define i64 @test_duplicated_phis(i64 noundef %N) {
; LEGACYPM-LABEL: define i64 @test_duplicated_phis
; LEGACYPM-SAME: (i64 noundef [[N:%.*]]) {
; LEGACYPM-NEXT:  entry:
; LEGACYPM-NEXT:    [[MUL:%.*]] = shl i64 [[N]], 1
; LEGACYPM-NEXT:    [[CMP6_NOT:%.*]] = icmp eq i64 [[MUL]], 0
; LEGACYPM-NEXT:    br i1 [[CMP6_NOT]], label [[FOR_END:%.*]], label [[FOR_BODY_PREHEADER:%.*]]
; LEGACYPM:       for.body.preheader:
; LEGACYPM-NEXT:    [[TMP0:%.*]] = icmp ult i64 [[MUL]], 4
; LEGACYPM-NEXT:    br i1 [[TMP0]], label [[FOR_END_LOOPEXIT_UNR_LCSSA:%.*]], label [[FOR_BODY_PREHEADER_NEW:%.*]]
; LEGACYPM:       for.body.preheader.new:
; LEGACYPM-NEXT:    [[UNROLL_ITER:%.*]] = and i64 [[MUL]], -4
; LEGACYPM-NEXT:    [[TMP1:%.*]] = add i64 [[UNROLL_ITER]], -4
; LEGACYPM-NEXT:    [[TMP2:%.*]] = lshr i64 [[TMP1]], 2
; LEGACYPM-NEXT:    [[TMP3:%.*]] = shl nuw nsw i64 [[TMP2]], 1
; LEGACYPM-NEXT:    [[TMP4:%.*]] = sub i64 -3, [[TMP3]]
; LEGACYPM-NEXT:    br label [[FOR_BODY:%.*]]
; LEGACYPM:       for.body:
; LEGACYPM-NEXT:    [[I_07:%.*]] = phi i64 [ 0, [[FOR_BODY_PREHEADER_NEW]] ], [ [[INC_3:%.*]], [[FOR_BODY]] ]
; LEGACYPM-NEXT:    [[INC_3]] = add i64 [[I_07]], 4
; LEGACYPM-NEXT:    [[NITER_NCMP_3_NOT:%.*]] = icmp eq i64 [[UNROLL_ITER]], [[INC_3]]
; LEGACYPM-NEXT:    br i1 [[NITER_NCMP_3_NOT]], label [[FOR_END_LOOPEXIT_UNR_LCSSA_LOOPEXIT:%.*]], label [[FOR_BODY]]
; LEGACYPM:       for.end.loopexit.unr-lcssa.loopexit:
; LEGACYPM-NEXT:    [[TMP5:%.*]] = add i64 [[TMP4]], 1
; LEGACYPM-NEXT:    br label [[FOR_END_LOOPEXIT_UNR_LCSSA]]
; LEGACYPM:       for.end.loopexit.unr-lcssa:
; LEGACYPM-NEXT:    [[RES_1_LCSSA_PH:%.*]] = phi i64 [ undef, [[FOR_BODY_PREHEADER]] ], [ [[TMP5]], [[FOR_END_LOOPEXIT_UNR_LCSSA_LOOPEXIT]] ]
; LEGACYPM-NEXT:    [[RES_09_UNR:%.*]] = phi i64 [ -1, [[FOR_BODY_PREHEADER]] ], [ [[TMP4]], [[FOR_END_LOOPEXIT_UNR_LCSSA_LOOPEXIT]] ]
; LEGACYPM-NEXT:    [[TMP6:%.*]] = and i64 [[N]], 1
; LEGACYPM-NEXT:    [[LCMP_MOD_NOT:%.*]] = icmp eq i64 [[TMP6]], 0
; LEGACYPM-NEXT:    [[SPEC_SELECT:%.*]] = select i1 [[LCMP_MOD_NOT]], i64 [[RES_1_LCSSA_PH]], i64 [[RES_09_UNR]]
; LEGACYPM-NEXT:    br label [[FOR_END]]
; LEGACYPM:       for.end:
; LEGACYPM-NEXT:    [[RES_0_LCSSA:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[SPEC_SELECT]], [[FOR_END_LOOPEXIT_UNR_LCSSA]] ]
; LEGACYPM-NEXT:    ret i64 [[RES_0_LCSSA]]
;
; NEWPM-LABEL: define i64 @test_duplicated_phis
; NEWPM-SAME: (i64 noundef [[N:%.*]]) {
; NEWPM-NEXT:  entry:
; NEWPM-NEXT:    [[MUL:%.*]] = shl i64 [[N]], 1
; NEWPM-NEXT:    [[CMP6_NOT:%.*]] = icmp eq i64 [[MUL]], 0
; NEWPM-NEXT:    br i1 [[CMP6_NOT]], label [[FOR_END:%.*]], label [[FOR_BODY_PREHEADER:%.*]]
; NEWPM:       for.body.preheader:
; NEWPM-NEXT:    [[TMP0:%.*]] = icmp ult i64 [[MUL]], 4
; NEWPM-NEXT:    br i1 [[TMP0]], label [[FOR_END_LOOPEXIT_UNR_LCSSA:%.*]], label [[FOR_BODY_PREHEADER_NEW:%.*]]
; NEWPM:       for.body.preheader.new:
; NEWPM-NEXT:    [[UNROLL_ITER:%.*]] = and i64 [[MUL]], -4
; NEWPM-NEXT:    br label [[FOR_BODY:%.*]]
; NEWPM:       for.body:
; NEWPM-NEXT:    [[LSR_IV:%.*]] = phi i64 [ [[LSR_IV_NEXT:%.*]], [[FOR_BODY]] ], [ 3, [[FOR_BODY_PREHEADER_NEW]] ]
; NEWPM-NEXT:    [[I_07:%.*]] = phi i64 [ 0, [[FOR_BODY_PREHEADER_NEW]] ], [ [[INC_3:%.*]], [[FOR_BODY]] ]
; NEWPM-NEXT:    [[INC_3]] = add i64 [[I_07]], 4
; NEWPM-NEXT:    [[LSR_IV_NEXT]] = add nsw i64 [[LSR_IV]], -2
; NEWPM-NEXT:    [[NITER_NCMP_3_NOT:%.*]] = icmp eq i64 [[UNROLL_ITER]], [[INC_3]]
; NEWPM-NEXT:    [[TMP1:%.*]] = add i64 [[LSR_IV_NEXT]], -3
; NEWPM-NEXT:    br i1 [[NITER_NCMP_3_NOT]], label [[FOR_END_LOOPEXIT_UNR_LCSSA_LOOPEXIT:%.*]], label [[FOR_BODY]]
; NEWPM:       for.end.loopexit.unr-lcssa.loopexit:
; NEWPM-NEXT:    [[REASS_SUB_LCSSA:%.*]] = phi i64 [ [[LSR_IV_NEXT]], [[FOR_BODY]] ]
; NEWPM-NEXT:    [[RES_1_3_LCSSA:%.*]] = phi i64 [ [[TMP1]], [[FOR_BODY]] ]
; NEWPM-NEXT:    [[TMP2:%.*]] = add i64 [[REASS_SUB_LCSSA]], -4
; NEWPM-NEXT:    br label [[FOR_END_LOOPEXIT_UNR_LCSSA]]
; NEWPM:       for.end.loopexit.unr-lcssa:
; NEWPM-NEXT:    [[RES_1_LCSSA_PH:%.*]] = phi i64 [ undef, [[FOR_BODY_PREHEADER]] ], [ [[RES_1_3_LCSSA]], [[FOR_END_LOOPEXIT_UNR_LCSSA_LOOPEXIT]] ]
; NEWPM-NEXT:    [[RES_09_UNR:%.*]] = phi i64 [ -1, [[FOR_BODY_PREHEADER]] ], [ [[TMP2]], [[FOR_END_LOOPEXIT_UNR_LCSSA_LOOPEXIT]] ]
; NEWPM-NEXT:    [[TMP3:%.*]] = and i64 [[N]], 1
; NEWPM-NEXT:    [[LCMP_MOD_NOT:%.*]] = icmp eq i64 [[TMP3]], 0
; NEWPM-NEXT:    [[SPEC_SELECT:%.*]] = select i1 [[LCMP_MOD_NOT]], i64 [[RES_1_LCSSA_PH]], i64 [[RES_09_UNR]]
; NEWPM-NEXT:    br label [[FOR_END]]
; NEWPM:       for.end:
; NEWPM-NEXT:    [[RES_0_LCSSA:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[SPEC_SELECT]], [[FOR_END_LOOPEXIT_UNR_LCSSA]] ]
; NEWPM-NEXT:    ret i64 [[RES_0_LCSSA]]
;
entry:
  %mul = shl i64 %N, 1
  %cmp6.not = icmp eq i64 %mul, 0
  br i1 %cmp6.not, label %for.end, label %for.body.preheader

for.body.preheader:
  %0 = icmp ult i64 %mul, 4
  br i1 %0, label %for.end.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:
  %unroll_iter = and i64 %mul, -4
  br label %for.body

for.body:
  %res.09 = phi i64 [ 0, %for.body.preheader.new ], [ %res.1.3, %for.body ]
  %i.07 = phi i64 [ 0, %for.body.preheader.new ], [ %inc.3, %for.body ]
  %niter = phi i64 [ 0, %for.body.preheader.new ], [ %niter.next.3, %for.body ]
  %res.1.1 = add i64 %res.09, -1
  %inc.1 = or disjoint i64 %i.07, 2
  %res.1.2 = add i64 %inc.1, %res.1.1
  %reass.sub = sub i64 %res.1.2, %i.07
  %res.1.3 = add i64 %reass.sub, -3
  %inc.3 = add nuw i64 %i.07, 4
  %niter.next.3 = add i64 %niter, 4
  %niter.ncmp.3.not = icmp eq i64 %niter.next.3, %unroll_iter
  br i1 %niter.ncmp.3.not, label %for.end.loopexit.unr-lcssa.loopexit, label %for.body

for.end.loopexit.unr-lcssa.loopexit:
  %1 = add i64 %reass.sub, -4
  br label %for.end.loopexit.unr-lcssa

for.end.loopexit.unr-lcssa:
  %res.1.lcssa.ph = phi i64 [ undef, %for.body.preheader ], [ %res.1.3, %for.end.loopexit.unr-lcssa.loopexit ]
  %res.09.unr = phi i64 [ -1, %for.body.preheader ], [ %1, %for.end.loopexit.unr-lcssa.loopexit ]
  %2 = and i64 %N, 1
  %lcmp.mod.not = icmp eq i64 %2, 0
  %spec.select = select i1 %lcmp.mod.not, i64 %res.1.lcssa.ph, i64 %res.09.unr
  br label %for.end

for.end:
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %spec.select, %for.end.loopexit.unr-lcssa ]
  ret i64 %res.0.lcssa
}
