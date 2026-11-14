/* GPT-2 split oracle for colibri-swift: upstream c/tok.h pretok_chunk_gpt2 with bpe_piece
 * replaced by a hook that records piece ends (tok_split.h is tok.h with that one change).
 * Prints the byte offsets where pieces end, one line per file argument. */
#include "tok_split.h"
int main(int argc, char **argv){
    for (int a = 1; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb"); if (!f) { perror(argv[a]); return 1; }
        fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        unsigned char *t = malloc(n+1); if (fread(t,1,n,f)!=(size_t)n) {} fclose(f);
        int *ends = malloc(sizeof(int)*(n+1)); int no = 0;
        pretok_chunk_gpt2(NULL, t, 0, (int)n, ends, &no, (int)n+1);
        for (int i=0;i<no;i++) printf("%s%d", i?" ":"", ends[i]);
        printf("\n"); free(ends); free(t);
    }
    return 0;
}
