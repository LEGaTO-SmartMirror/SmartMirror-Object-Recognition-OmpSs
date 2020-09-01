#ifndef COMMON_H
#define COMMON_H

#define OPENCV 1

#include <darknet.h>
#include "utils.h"
#include <sys/time.h>
#include "option_list.h"
#include "parser.h"
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include "MYSort.h"
#include <nanos6/debug.h>
#include <nanos6/cluster.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#define SERIAL
//#define D

#define _assert(cond)                                           \
{                                                               \
    if(!cond)                                                   \
    {                                                           \
        fprintf(stderr, "[%s][%s]:[%u] Assertion '%s' failed. \n",    \
        __func__, __FILE__, __LINE__, #cond);                   \
        abort();                                                \
    }                                                           \
}

#define dump(v) printf("%s",#v);

static inline void *_dmalloc(size_t size)
{
    #ifdef SERIAL
    void *ret = malloc(size);
    #else
    void *ret =  nanos6_dmalloc(size, nanos6_equpart_distribution, 0, NULL);
    #endif
    if (!ret)
    {    
        fprintf(stderr,"Could not allocate Nanos6 distributed memory\n");     
        dump(size);     
        exit(1);
    }    

    return ret; 
}
static inline void *_malloc(size_t size, const char* objName)
{
    

    void *ret = malloc(size);
     
    if (!ret)
    {
        perror("_lmalloc()");
        exit(1);
    }
    #ifdef D
//    printf("%s _malloced with size %2.2f KB. \n", objName, size*10e-3);
    printf("%s malloced with size %ld  Addr %p - %p \n", objName, size/**10e-3*/, (void *)objName, (void *)objName+size);

    #endif 
    return ret;
}

static inline void *_lmalloc(size_t size, const char* objName)
{
    

    #ifdef SERIAL
    void *ret = malloc(size);
    #else
    void *ret = nanos6_lmalloc(size);
    #endif  
    if (!ret)
    {
        perror("nanos6_lmalloc()");
        exit(1);
    }
    #ifdef D
    printf("%s lmalloced with size %ld  Addr %p - %p \n", objName, size/**10e-3*/, (void *)objName, (void *)objName+size);
    #endif 

    return ret;
}


static inline void _lfree(void *ptr, size_t size)
{
    _assert(ptr);
    nanos6_lfree(ptr, size);
}

static inline void _free_detections(detection *dets, int n)
{
    //_assert(dets);
   
    if(n <= 0)
        return;

    for (int i = 0; i < n; ++i) 
    {
         _assert(dets[i].prob);
        free(dets[i].prob);
        if (dets[i].uc) free(dets[i].uc);
        if (dets[i].mask) free(dets[i].mask);
    }
    //printf("about to free dets.\n");
    free(dets);
}


void _copy_detections(detection *src, detection *dest, int nboxes, int classes)
{
    
    printf("I am copying: nboxes: %d , and classes %d\n", nboxes, classes);
    //asuming **num** paramter is alwayes 0
   // int nboxes = num_detections(net, thresh);
   _assert(src);
   _assert(dest);

    for(int i = 0; i < nboxes; ++i)
    {
        dest[i].bbox.x = src[i].bbox.x;
        dest[i].bbox.y = src[i].bbox.y;
        dest[i].bbox.w = src[i].bbox.w;
        dest[i].bbox.h = src[i].bbox.h;
        dest[i].classes = src[i].classes;
        dest[i].objectness = src[i].objectness;
        dest[i].sort_class = src[i].sort_class;
        dest[i].points = src[i].points;
    
        if(src[i].prob)     memcpy(src[i].prob, dest[i].prob, classes * sizeof(float));
        else                dest[i].prob = NULL; 
    
        if(src[i].mask)     memcpy(src[i].mask, dest[i].mask, classes * sizeof(float));
        else                dest[i].mask = NULL;
    
        if(src[i].uc)       memcpy(src[i].uc, dest[i].uc, classes * sizeof(float));
        else                dest[i].uc = NULL;
    }
}

detection *_nanos6_calloc_detections(int nboxes, int classes, detection * _dets)
{
   // detection *_dets = (detection *)nanos6_lmalloc(nboxes * sizeof(detection));
    //_assert(_dets);
    /*
    size_t _allocSize = classes * sizeof(float);
    
    for(int i = 0; i < nboxes; ++i)
    {
        _dets[i].bbox.x = 0.0f;
        _dets[i].bbox.y = 0.0f;
        _dets[i].bbox.w = 0.0f;
        _dets[i].bbox.h = 0.0f;
        _dets[i].classes = 0;
        _dets[i].objectness = 0.0f;
        _dets[i].sort_class = 0;
        _dets[i].points = 0;

        _dets[i].prob = (float *)_lmalloc(_allocSize);
        _assert(_dets[i].prob);
        memset(_dets[i].prob, 0, _allocSize);

        _dets[i].mask = (float *)_lmalloc(_allocSize);
        _assert(_dets[i].mask);
        memset(_dets[i].prob, 0, _allocSize);

        _dets[i].uc = (float *)_lmalloc(_allocSize);
        _assert(_dets[i].uc);
        memset(_dets[i].prob, 0, _allocSize);
    }
    */
    return _dets;    
}

void _print_hostname(const char *s, ...)
{
    va_list args;
    char hostname[1024];
    char buffer[1024];
    gethostname(hostname, 1024);
    va_start(args, s);
    vsprintf(buffer, s, args);
    printf("on %s: %s \n", hostname, buffer);
    va_end(args);
}
void _scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}
void _forward_network(network net, network_state state)
{
    printf("B0\n");
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta && state.train){
            _scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        printf("B0.1\n");
        l.forward(l, state);
        printf("B0.2\n");

        state.input = l.output;
    }
    printf("B1\n");

}
float *_network_predict(network net, float *input)
{
#ifdef GPU
    if(gpu_index >= 0)  return network_predict_gpu(net, input);
#endif

    network_state state = {0};
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    printf("A0\n");
    _forward_network(net, state);
    printf("A1\n");
    float *out = get_network_output(net);
    printf("A2\n");
    return out;
}

float *_network_predict_image(network *net, image im)
{
    printf("P0\n"); 
    float *p;
    if(net->batch != 1) set_batch_network(net, 1);
    printf("P1\n");
    if (im.w == net->w && im.h == net->h) {
        // Input image is the same size as our net, predict on that image
        printf("P1.1\n");
       // _printImageData(im);
        p = network_predict(*net, im.data);
    }
    else {
        // Need to resize image to the desired size for the net
        printf("P1.2\n im.w %d im.h %d ||| net.w %d net.h %d \n", im.w,im.h, net->w, net->h);
        image imr = resize_image(im, net->w, net->h);
        printf("P1.3\n imr.w*imr.h*imr.c %d %d %d %d \n net->w %d, net->h %d", 
        imr.w,imr.h,imr.c, imr.w*imr.h*imr.c, net->w, net->h);
        _assert(imr.data);
        _assert(net);
        for(int i = 0; i < imr.w*imr.h; ++i) {
          //  if(imr.data[i] ) printf("%f ", imr.data[i]);
        }
        printf("\n %d ", net->n);            
        _print_hostname(" P1.3.0");
        p = network_predict(*net, imr.data);
        printf("P1.4\n");
        free_image(imr);
        printf("P1.5\n");
    }
    printf("P2\n");
    return p;
}

void _printImageData(const image img)
{
    _assert(&img);
    for(int i = 0; i < img.h*img.w*img.c ; ++i) {
        //if(img.data[i]) 
            printf("%f ", img.data[i]);
    }
}

void _print_array(float *array, int length, const char *name) 
{
    printf("%s:  ", name);
    for(int i = 0; i < length; ++i)
    {
        printf("%0.2f, ", array[i]);
    }
    printf(".\n");

}

/*
typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}
list *make_list()
{
    list* l = (list*)xmalloc(sizeof(list));
    l->size = 0;
    //l->front = 0;
    //memset(l->front, 0, sizeof(node));
    l->back = 0;
    return l;
}*/
#endif // COMMON_H