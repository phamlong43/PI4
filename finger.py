#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Hoang Duc Luong");
MODULE_DESCRIPTION("Module kernel tinh vector, ma tran va chinh hop");
MODULE_VERSION("1.0");

#define SIZE 3  // kich thuoc vi du cho vector va ma tran

// Ham tinh tich vo huong cua hai vector
int tinh_tich_vo_huong(int *v1, int *v2, int size) {
    int ket_qua = 0;
    int i;
    for (i = 0; i < size; i++) {
        ket_qua += v1[i] * v2[i];
    }
    return ket_qua;
}

// Ham tinh tong hai ma tran
void tinh_tong_ma_tran(int m1[SIZE][SIZE], int m2[SIZE][SIZE], int ket_qua[SIZE][SIZE]) {
    int i, j;
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            ket_qua[i][j] = m1[i][j] + m2[i][j];
        }
    }
}

// Ham tinh hieu hai ma tran
void tinh_hieu_ma_tran(int m1[SIZE][SIZE], int m2[SIZE][SIZE], int ket_qua[SIZE][SIZE]) {
    int i, j;
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            ket_qua[i][j] = m1[i][j] - m2[i][j];
        }
    }
}

// Ham tinh tich hai ma tran
void tinh_tich_ma_tran(int m1[SIZE][SIZE], int m2[SIZE][SIZE], int ket_qua[SIZE][SIZE]) {
    int i, j, k;
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            ket_qua[i][j] = 0;
            for (k = 0; k < SIZE; k++) {
                ket_qua[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

// Ham tinh giai thua
unsigned long tinh_giai_thua(int n) {
    unsigned long ket_qua = 1;
    int i;
    for (i = 1; i <= n; i++) {
        ket_qua *= i;
    }
    return ket_qua;
}

// Ham tinh chinh hop A(n, k)
unsigned long tinh_chinh_hop(int n, int k) {
    return tinh_giai_thua(n) / tinh_giai_thua(n - k);
}

static int __init bai3_1_init(void) {
    printk(KERN_INFO "[bai3.1] Module duoc nap vao kernel\n");

    int v1[SIZE] = {1, 2, 3};
    int v2[SIZE] = {4, 5, 6};
    int m1[SIZE][SIZE] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int m2[SIZE][SIZE] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int ket_qua_ma_tran[SIZE][SIZE];

    int tich_vo_huong = tinh_tich_vo_huong(v1, v2, SIZE);
    printk(KERN_INFO "Tich vo huong: %d\n", tich_vo_huong);

    tinh_tong_ma_tran(m1, m2, ket_qua_ma_tran);
    printk(KERN_INFO "Tong ma tran:\n");
    int i, j;
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            printk(KERN_INFO "%d ", ket_qua_ma_tran[i][j]);
        }
        printk(KERN_INFO "\n");
    }

    tinh_hieu_ma_tran(m1, m2, ket_qua_ma_tran);
    printk(KERN_INFO "Hieu ma tran:\n");
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            printk(KERN_INFO "%d ", ket_qua_ma_tran[i][j]);
        }
        printk(KERN_INFO "\n");
    }

    tinh_tich_ma_tran(m1, m2, ket_qua_ma_tran);
    printk(KERN_INFO "Tich ma tran:\n");
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            printk(KERN_INFO "%d ", ket_qua_ma_tran[i][j]);
        }
        printk(KERN_INFO "\n");
    }

    unsigned long chinh_hop = tinh_chinh_hop(5, 2);
    printk(KERN_INFO "Chinh hop A(5, 2): %lu\n", chinh_hop);

    return 0;
}

static void __exit bai3_1_exit(void) {
    printk(KERN_INFO "[bai3.1] Module da duoc go khoi kernel\n");
}

module_init(bai3_1_init);
module_exit(bai3_1_exit);
