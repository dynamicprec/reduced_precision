/*BEGIN_LEGAL
Intel Open Source License

Copyright (c) 2002-2018 Intel Corporation. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.  Redistributions
in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.  Neither the name of
the Intel Corporation nor the names of its contributors may be used to
endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE INTEL OR
ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
END_LEGAL */

#include "dynamic.hpp"

// define STATS_ON to update stats
uint64_t num_WU_insts = 0;
uint64_t num_vml_insts = 0;
uint64_t num_fp_insts = 0;
uint64_t num_fma_insts = 0;
uint64_t num_total_insts = 0;
uint64_t num_instrumented_regs = 0;
uint64_t num_src_reg_calls = 0;
uint64_t num_dest_reg_calls = 0;

#ifdef AVX512
__m512 InstrumentationSourceOperand64(PIN_REGISTER *operand1)
{
#ifdef STATS_ON
    num_src_reg_calls++;
#endif
    return ToBFloatTensorFlowVect64((__m512*) & (operand1->flt[0]));
}
#endif

// Instrumentation routines for source registers
// Returning an __m256 type avoids vzeroupper
__m256 InstrumentationSourceOperand32(PIN_REGISTER *operand1)
{
#ifdef STATS_ON
    num_src_reg_calls++;
#endif
    return ToBFloatTensorFlowVect32((__m256*) & (operand1->flt[0]));
}


__m128 InstrumentationSourceOperand16(PIN_REGISTER *operand1)
{
#ifdef STATS_ON
    num_src_reg_calls++;
#endif
    return ToBFloatTensorFlowVect16((__m128*) & (operand1->flt[0]));
}


void InstrumentationSourceOperand8(PIN_REGISTER *operand1)
{
#ifdef STATS_ON
    num_src_reg_calls++;
#endif
    operand1->flt[0] = ToBFloatTensorFlow(operand1->flt[0]);
}

#ifdef AVX512
__m512 InstrumentationDestOperand64(PIN_REGISTER *operand1)
{
#ifdef STATS_ON
    num_dest_reg_calls++;
#endif
    return ToBFloatTensorFlowVect64((__m512*) & (operand1->flt[0]));
}
#endif

// Instrumentation routines for destination registers
__m256 InstrumentationDestOperand32(PIN_REGISTER *operand1)
{
#ifdef STATS_ON
    num_dest_reg_calls++;
#endif
    return ToBFloatTensorFlowVect32((__m256*) & (operand1->flt[0]));
}

__m128 InstrumentationDestOperand16(PIN_REGISTER *operand1)
{
#ifdef STATS_ON
    num_dest_reg_calls++;
#endif
    return ToBFloatTensorFlowVect16((__m128*) & (operand1->flt[0]));
}


void InstrumentationDestOperand8(PIN_REGISTER *operand1)
{
#ifdef STATS_ON
    num_dest_reg_calls++;
#endif
    operand1->flt[0] = ToBFloatTensorFlow(operand1->flt[0]);
}


// Instrumentation routines for STATS
VOID CountAllInstructions(UINT32 num)
{
    num_total_insts += num;
}

VOID CountFPInstructions(UINT32 num)
{
    num_fp_insts += num;
}

VOID CountFMAInstructions(UINT32 num)
{
    num_fma_insts += num;
}

VOID CountVMLInstructions(UINT32 num)
{
    num_vml_insts += num;
}

VOID CountWUInstructions(UINT32 num)
{
    num_WU_insts += num;
}

// Initial mode in FP32 precision
VOID Trace_Native(TRACE trace, VOID *v){
    PIN_Detach();
}

// Pin calls this function every time a new trace (single entry, multiple exits)
// is executed, after the first detach
VOID Trace(TRACE trace, VOID *v)
{
    std::set<REG> truncated_regs;
    string routineName;
#ifndef ROUTINES_ON
    UINT32 numOperands, size;
    unsigned int i, num_src_op = 0, num_dest_op = 0;
    int src_ids[4] = {0};
    int dest_ids[1] = {0};
#endif

    truncated_regs.clear();

    // Visit every basic block in the trace
    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl))
    {

        // Count all instructions
#ifdef STATS_ON
        BBL_InsertCall(bbl, IPOINT_ANYWHERE, (AFUNPTR)CountAllInstructions,
                IARG_UINT32, BBL_NumIns(bbl), IARG_END);
#endif

        // For each instruction of the BBL
        for (INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins))
        {
            //First check routine
            RTN routine = INS_Rtn(ins);
            if (RTN_Valid(routine))
                routineName = RTN_Name(routine);
            else
                routineName = "unknown";

#ifdef ROUTINES_ON
            *out << routineName << " " << INS_Mnemonic(ins) << endl;
#endif

#ifndef ROUTINES_ON
            // Never instrument vml routines
            if ((routineName.find("mkl_vml") != string::npos))
            {
#ifdef STATS_ON

                INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)CountVMLInstructions,
                        IARG_UINT32, 1, IARG_END);
#endif
                continue;
            }

            // If the calculation need to be done in full bf16 don't avoid
            // BN or WU routines
            if ((mode == FMA_MP_FP32_WU_BN) || (mode == FMA_BF16_FP32_WU_BN))
            {
                // Never instrument saxpy or saxpby routines used in weight updates and
                // SGDFusion
                if ((routineName.find("saxpy") != string::npos) ||
                        (routineName.find("axpy_axpby") != string::npos) ||
                        (routineName.find("saxpby") != string::npos) ||
                        (routineName.find("addcdiv") != string::npos) ||
                        (routineName.find("add_") != string::npos))
                {
#ifdef STATS_ON
                    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)CountWUInstructions,
                            IARG_UINT32, 1, IARG_END);
#endif
                    continue;
                }
            }//closes mode == FMA_MP-FP32_WU_BN
            else
            {
                if ((mode == FMA_MP_FP32_WU) || (mode == FMA_BF16_FP32_WU))
                {
                    // Never instrument saxpy or saxpby routines used in weight updates and
                    // SGDFusion
                    if ((routineName.find("saxpy") != string::npos) ||
                        (routineName.find("addcdiv") != string::npos) ||
                      (routineName.find("add_") != string::npos))
                    {
#ifdef STATS_ON
                        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)CountWUInstructions,
                                IARG_UINT32, 1, IARG_END);
#endif
                        continue;
                    }
                } //closes mode == FMA_MP-FP32_WU
            }
            OPCODE iclass = INS_Opcode(ins);
#ifdef STATS_ON
            if(isFP(iclass))
            {
                INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)CountFPInstructions,
                        IARG_UINT32, 1, IARG_END);

            }
#endif
            // Check if it needs to be instrumented (is FMA)
            if (isFMA(iclass))
            {
                // Count all FMA instructions
#ifdef STATS_ON
                INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)CountFMAInstructions,
                        IARG_UINT32, 1, IARG_END);
#endif
                numOperands = INS_OperandCount(ins);
                // For all operands
                for (i = 0; i < numOperands; i++)
                {
                    // If it is register and source and not already in truncated set,
                    // register for truncation IBEFORE
                    if (INS_OperandIsReg(ins, i) && INS_OperandRead(ins, i)
                            && !truncated_regs.count(INS_OperandReg(ins, i)))
                    {
                        if ((mode == FMA_MP) || (mode == FMA_MP_FP32_WU) || (mode == FMA_MP_FP32_WU_BN))
                        {
                            // Check FMA types to avoid truncation of acumulator src reg
                            if (isVFMA132(iclass) && i == numOperands - 2) continue;
                            if (isVFMA213(iclass) && i == numOperands - 1) continue;
                            if (isVFMA231(iclass) && i == 0) continue;
                        }
                        // Will be instrumented for truncation
                        src_ids[num_src_op] = i;
                        num_src_op++;
                        truncated_regs.insert(INS_OperandReg(ins, i));
                        //assert(num_src_op < 4);
                    }
                    // If it is register and destination, register for truncation IAFTER
                    if (INS_OperandIsReg(ins, i) && INS_OperandWritten(ins, i))
                    {
                        if (((mode == FMA_MP) || (mode == FMA_MP_FP32_WU) || (mode == FMA_MP_FP32_WU_BN)) && isFMA(iclass))
                        {
                            // Register about the be written and not truncated, remove from set
                            truncated_regs.erase(INS_OperandReg(ins, i));
                        }
                        else
                        {
                            // Will be instrumented for truncation
                            dest_ids[num_dest_op] = i;
                            num_dest_op++;
                            truncated_regs.insert(INS_OperandReg(ins, i));
                            assert(num_dest_op < 2);
                        }
                    }
                }
#ifdef STATS_ON
                num_instrumented_regs += num_src_op;
                num_instrumented_regs += num_dest_op;
#endif
                for (i = 0; i < num_src_op; i++)
                {
                    REG operand = INS_OperandReg(ins, src_ids[i]);
                    size = REG_Size(operand);
                    assert(size == 32 || size == 16 || size == 64 || size == 8);
                    switch (size)
                    {
                        case 64:
                            INS_InsertCall(ins, IPOINT_BEFORE,
                                    (AFUNPTR)InstrumentationSourceOperand64,
                                    IARG_REG_REFERENCE, INS_OperandReg(ins, src_ids[i]),
                                    IARG_END);
                            break;
                        case 32:
                            INS_InsertCall(ins, IPOINT_BEFORE,
                                    (AFUNPTR)InstrumentationSourceOperand32,
                                    IARG_REG_REFERENCE, INS_OperandReg(ins, src_ids[i]),
                                    IARG_END);
                            break;
                        case 16:
                            INS_InsertCall(ins, IPOINT_BEFORE,
                                    (AFUNPTR)InstrumentationSourceOperand16,
                                    IARG_REG_REFERENCE, INS_OperandReg(ins, src_ids[i]),
                                    IARG_END);
                            break;
                        case 8:
                            //INS_InsertCall(ins, IPOINT_BEFORE,
                            //(AFUNPTR)InstrumentationSourceOperand8,
                            //IARG_REG_REFERENCE, INS_OperandReg(ins,src_ids[i]),
                            //IARG_END);
                            break;
                        default:
                            break;
                    }
                }

                for (i = 0; i < num_dest_op; i++)
                {
                    REG operand = INS_OperandReg(ins, dest_ids[i]);
                    size = REG_Size(operand);
                    assert(size == 32 || size == 16 || size == 64 || size == 8);
                    switch (size)
                    {
                        case 64:
                            INS_InsertCall(ins, IPOINT_AFTER,
                                    (AFUNPTR)InstrumentationDestOperand64,
                                    IARG_REG_REFERENCE, INS_OperandReg(ins, dest_ids[i]),
                                    IARG_END);
                            break;
                        case 32:
                            INS_InsertCall(ins, IPOINT_AFTER,
                                    (AFUNPTR)InstrumentationDestOperand32,
                                    IARG_REG_REFERENCE, INS_OperandReg(ins, dest_ids[i]),
                                    IARG_END);
                            break;
                        case 16:
                            INS_InsertCall(ins, IPOINT_AFTER,
                                    (AFUNPTR)InstrumentationDestOperand16,
                                    IARG_REG_REFERENCE, INS_OperandReg(ins, dest_ids[i]),
                                    IARG_END);
                            break;
                        case 8:
                            break;
                        default:
                            break;
                    }
                }
            }//closes if(isFMA)
            else
            {
                // Due to the optimization to avoid truncating regs that are already truncated,
                // we need to check all instructions for destination registers and
                // remove from tuncated list if present
                numOperands = INS_OperandCount(ins);
                // For all operands
                for (i = 0; i < numOperands; i++)
                {
                    // If it is register and destination - check if in set and remove
                    if (INS_OperandIsReg(ins, i) && INS_OperandWritten(ins, i))
                    {
                        truncated_regs.erase(INS_OperandReg(ins, i));
                    }
                }
            }

            num_dest_op = 0;
            num_src_op = 0;
            memset(src_ids, 0, sizeof(src_ids));
            memset(dest_ids, 0, sizeof(dest_ids));
#endif //#ifdef ROUTINES_ON
        }
        // Clear list for next BBL
        truncated_regs.clear();
    }
}

void print_stats()
{
    *out << "Statistics:" << endl;
    *out << "\tnum_WU_insts: " << num_WU_insts << endl;
    *out << "\tnum_vml_insts: " << num_vml_insts << endl;
    *out << "\tnum_fp_insts: " << num_fp_insts << endl;
    *out << "\tnum_fma_insts: " << num_fma_insts << endl;
    *out << "\tnum_total_insts: " << num_total_insts << endl;
    *out << "\tnum_instrumented_regs: " << num_instrumented_regs << endl;
    *out << "\tnum_src_reg_calls: " << num_src_reg_calls << endl;
    *out << "\tnum_dest_reg_calls: " << num_dest_reg_calls << endl;
}

// This function is called when the application exits
// It prints the name and count for each procedure
VOID Fini(INT32 code, VOID *v)
{
#ifdef STATS_ON
    print_stats();
#endif
    *out << "End of the PinTool" << endl;
}

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */

INT32 Usage()
{
    cerr << "This Pintool truncate FP32 to BF16 on some Floating Point Instructions" << endl;
    cerr << endl << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}


// Call back function called when the application it is started
VOID ApplicationStart(VOID *arg)
{
    *out << "Application start ..." << endl;
    sleep(1);
}

// Call back function called when the PinTool it is going to be
// detached from the application. Permits avoid the use of time
// sleep on the python code
VOID DetachCompleted(VOID *arg)
{
    *out << "Detached completed ..." << endl;
    sleep(1);
}

// Callback function executed when the PinTool it is going to be
// attached to the application.
VOID attachMain(VOID* arg)
{
    TRACE_AddInstrumentFunction(Trace, 0);
    PIN_AddApplicationStartFunction(ApplicationStart, 0);
    PIN_AddDetachFunction(DetachCompleted, 0);
}

// Thread function to re-attach the PIN tool
static VOID attachThread(VOID *arg)
{
    int fd;
    char c;
//    char *fifo_name = (char*)arg;

    *out << "Thread Launched..." << endl;
    while (1)
    {
        *out << "Waiting in fifopipe" << endl;
        fd = open("fifopipe", O_RDONLY);
        read(fd, &c, 1);
        *out << "Received: " << c << endl;
        switch (c)
        {
        case 'A':
            mode = FMA_BF16;
            break;
        case 'B':
            mode = FMA_MP_FP32_WU_BN;
            break;
        case 'C':
            mode = FMA_MP_FP32_WU;
            break;
        case 'D':
            mode = FMA_MP;
            break;
        case 'E':
            mode = FMA_BF16_FP32_WU_BN;
            break;
        case 'F':
            mode = FMA_BF16_FP32_WU;
            break;
        default:
            mode = NATIVE;
            close(fd);
            PIN_Detach();
            sleep(4);
            *out << "Detached ..." << endl;
            continue;
        }
        close(fd);
        // First Detach
        PIN_Detach();
        sleep(4);
        *out << "Detached ..." << endl;
        while (ATTACH_FAILED_DETACH == PIN_Attach(attachMain, (VOID*)0))
        {
            *out << "Attach Failed " << mode << endl;
            exit(-1);
        }
        *out << "Attached " << mode << endl;
    }
}

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
    "o", "new_tests.out", "specify output file name");

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */
int main(int argc, char * argv[])
{
    // Initialize symbol table code, needed for rtn instrumentation
    PIN_InitSymbols();
    scratchReg = PIN_ClaimToolRegister();
    if (!REG_valid(scratchReg))
    {
        std::cerr << "Cannot allocate a scratch register.\n";
        std::cerr << std::flush;
        return 1;
    }

    // Initialize pin
    if (PIN_Init(argc, argv)) return Usage();
    // Initial mode baseline FMA_MP_FP32_WU_BN

    stringstream ss(KnobOutputFile.Value());
    string token;
    getline(ss, token,',');
    string filename = token;
    getline(ss, token, ',');
    string pipe_name = token;
    out = new ofstream(filename.c_str());

    mode = NATIVE;
    //mode = FMA_MP_FP32_WU_BN;
    // Register Routine to be called to instrument rtn
    // if you set mode = NATIVE at the begining, you
    // need to set as first parameter Trace_Native
    // otherwise just call Trace
    TRACE_AddInstrumentFunction(Trace_Native, 0);
    PIN_AddApplicationStartFunction(ApplicationStart, 0);
    PIN_AddDetachFunction(DetachCompleted, 0);

    // Register Fini to be called when the application exits
    PIN_AddFiniFunction(Fini, 0);

    // Create a thread inside the PIN Tool to do the re-attach
    // process
    THREADID tid = PIN_SpawnInternalThread(attachThread, (void*)pipe_name.c_str(), 0x40000, NULL);
    assert(tid != INVALID_THREADID);

    // Start the program, never returns
    PIN_StartProgram();
    return 0;
}
