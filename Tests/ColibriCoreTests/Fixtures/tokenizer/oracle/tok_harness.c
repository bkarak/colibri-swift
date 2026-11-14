/* Tokenizer oracle for colibri-swift: loads the GGUF tokenizer with upstream qwen36.c's
 * load_tokenizer_gguf and prints encode_text's ids for each file argument, one line per
 * file. EXACT=1 sets g_tok_exact (full Unicode classes + NFC, upstream's HF-exact mode). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define main qwen36_main_unused
#include "qwen36.c"
#undef main
int main(int argc, char **argv){
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf file...\n", argv[0]); return 2; }
    if (gguf_open(&g_gg, argv[1])) { fprintf(stderr, "gguf_open: %s\n", g_gg.err); return 1; }
    if (load_tokenizer_gguf(&g_gg)) return 1;
    if (getenv("EXACT")) g_tok_exact = 1;
    for (int a = 2; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb"); if (!f) { perror(argv[a]); return 1; }
        fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        char *t = malloc(n+1); if (fread(t,1,n,f)!=(size_t)n) {} t[n]=0; fclose(f);
        int *ids=NULL, k=0; encode_text(t, &ids, &k);
        for (int i=0;i<k;i++) printf("%s%d", i?" ":"", ids[i]);
        printf("\n"); free(ids); free(t);
    }
    return 0;
}
