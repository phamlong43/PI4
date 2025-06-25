#include <linux/init.h>
#include <linux/module.h>
#include <linux/device.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/string.h>
#include <linux/slab.h>  // for kmalloc and kfree

#define DEVICE_NAME "lab6"
#define CLASS_NAME "crypto"
#define BUFFER_SIZE 1024

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Cryptographic Character Driver using ioctl");
MODULE_VERSION("1.0");

static int major_number;
static char message[BUFFER_SIZE] = {0};
static char encrypted[BUFFER_SIZE] = {0};
static struct class* crypto_class = NULL;
static struct device* crypto_device = NULL;

// IOCTL commands
#define CRYPTO_IOC_MAGIC 'c'
#define CRYPTO_SHIFT_ENCRYPT         _IOW(CRYPTO_IOC_MAGIC, 1, int)
#define CRYPTO_SHIFT_DECRYPT         _IOW(CRYPTO_IOC_MAGIC, 2, int)
#define CRYPTO_SUBSTITUTION_ENCRYPT _IO(CRYPTO_IOC_MAGIC, 3)
#define CRYPTO_SUBSTITUTION_DECRYPT _IO(CRYPTO_IOC_MAGIC, 4)
#define CRYPTO_TRANSPOSITION_ENCRYPT _IOW(CRYPTO_IOC_MAGIC, 5, int)
#define CRYPTO_TRANSPOSITION_DECRYPT _IOW(CRYPTO_IOC_MAGIC, 6, int)

// =========================
// === Cipher Functions ===
// =========================

static void shift_encrypt(char* text, int shift) {
    int i;
    shift %= 26;
    for (i = 0; text[i]; i++) {
        if (text[i] >= 'a' && text[i] <= 'z')
            text[i] = ((text[i] - 'a' + shift) % 26) + 'a';
        else if (text[i] >= 'A' && text[i] <= 'Z')
            text[i] = ((text[i] - 'A' + shift) % 26) + 'A';
    }
}

static void shift_decrypt(char* text, int shift) {
    shift_encrypt(text, 26 - (shift % 26));
}

static const char sub_key[27] = "zyxwvutsrqponmlkjihgfedcba";

static void substitution_encrypt(char* text) {
    int i;
    for (i = 0; text[i]; i++) {
        if (text[i] >= 'a' && text[i] <= 'z')
            text[i] = sub_key[text[i] - 'a'];
        else if (text[i] >= 'A' && text[i] <= 'Z')
            text[i] = sub_key[text[i] - 'A'] - 32;
    }
}

static void substitution_decrypt(char* text) {
    int i, j;
    for (i = 0; text[i]; i++) {
        if (text[i] >= 'a' && text[i] <= 'z') {
            for (j = 0; j < 26; j++) {
                if (sub_key[j] == text[i]) {
                    text[i] = 'a' + j;
                    break;
                }
            }
        } else if (text[i] >= 'A' && text[i] <= 'Z') {
            char lower = text[i] + 32;
            for (j = 0; j < 26; j++) {
                if (sub_key[j] == lower) {
                    text[i] = 'A' + j;
                    break;
                }
            }
        }
    }
}

static void transposition_encrypt(char* text, int key) {
    int len = strlen(text);
    if (len == 0 || key <= 1) return;

    int rows = (len + key - 1) / key;
    char *temp = kmalloc(len + 1, GFP_KERNEL);
    int i, j, idx = 0;
    if (!temp) return;

    for (j = 0; j < key; j++) {
        for (i = 0; i < rows; i++) {
            int pos = i * key + j;
            if (pos < len) {
                temp[idx++] = text[pos];
            }
        }
    }

    temp[len] = '\0';
    strscpy(text, temp, len + 1);
    kfree(temp);
}

static void transposition_decrypt(char* text, int key) {
    int len = strlen(text);
    if (len == 0 || key <= 1) return;

    int rows = (len + key - 1) / key;
    char *temp = kmalloc(len + 1, GFP_KERNEL);
    int full_columns = len % key;
    int i, j, idx = 0;

    if (!temp) return;
    if (full_columns == 0) full_columns = key;

    for (j = 0; j < key; j++) {
        int col_size = (j < full_columns) ? rows : rows - 1;
        for (i = 0; i < col_size; i++) {
            int pos = i * key + j;
            if (pos < len && idx < len) {
                temp[pos] = text[idx++];
            }
        }
    }

    temp[len] = '\0';
    strscpy(text, temp, len + 1);
    kfree(temp);
}

// =========================
// === File Operations  ===
// =========================

static int dev_open(struct inode *inodep, struct file *filep) {
    printk(KERN_INFO "Crypto: Device opened\n");
    return 0;
}

static int dev_release(struct inode *inodep, struct file *filep) {
    printk(KERN_INFO "Crypto: Device closed\n");
    return 0;
}

static ssize_t dev_read(struct file *filep, char *buffer, size_t len, loff_t *offset) {
    size_t msg_len = strlen(encrypted);
    size_t remaining = msg_len - *offset;
    size_t to_copy = (len < remaining) ? len : remaining;

    if (*offset >= msg_len)
        return 0;

    if (copy_to_user(buffer, encrypted + *offset, to_copy))
        return -EFAULT;

    *offset += to_copy;
    return to_copy;
}

static ssize_t dev_write(struct file *filep, const char *buffer, size_t len, loff_t *offset) {
    if (len >= BUFFER_SIZE)
        len = BUFFER_SIZE - 1;

    if (copy_from_user(message, buffer, len))
        return -EFAULT;

    message[len] = '\0';
    strcpy(encrypted, message);

    *offset = 0;
    filep->f_pos = 0;
    return len;
}

static long dev_ioctl(struct file *filep, unsigned int cmd, unsigned long arg) {
    int key;
    strscpy(encrypted, message, BUFFER_SIZE);

    switch (cmd) {
        case CRYPTO_SHIFT_ENCRYPT:
            if (copy_from_user(&key, (int *)arg, sizeof(int))) return -EFAULT;
            shift_encrypt(encrypted, key);
            break;
        case CRYPTO_SHIFT_DECRYPT:
            if (copy_from_user(&key, (int *)arg, sizeof(int))) return -EFAULT;
            shift_decrypt(encrypted, key);
            break;
        case CRYPTO_SUBSTITUTION_ENCRYPT:
            substitution_encrypt(encrypted);
            break;
        case CRYPTO_SUBSTITUTION_DECRYPT:
            substitution_decrypt(encrypted);
            break;
        case CRYPTO_TRANSPOSITION_ENCRYPT:
            if (copy_from_user(&key, (int *)arg, sizeof(int))) return -EFAULT;
            transposition_encrypt(encrypted, key);
            break;
        case CRYPTO_TRANSPOSITION_DECRYPT:
            if (copy_from_user(&key, (int *)arg, sizeof(int))) return -EFAULT;
            transposition_decrypt(encrypted, key);
            break;
        default:
            return -EINVAL;
    }

    filep->f_pos = 0;
    return 0;
}

static struct file_operations fops = {
    .open = dev_open,
    .read = dev_read,
    .write = dev_write,
    .release = dev_release,
    .unlocked_ioctl = dev_ioctl,
};

// ==============================
// === Module Init / Exit ======
// ==============================

static int __init crypto_init(void) {
    printk(KERN_INFO "Crypto: Initializing module...\n");

    major_number = register_chrdev(0, DEVICE_NAME, &fops);
    if (major_number < 0)
        return major_number;

    crypto_class = class_create(CLASS_NAME);
    if (IS_ERR(crypto_class)) {
        unregister_chrdev(major_number, DEVICE_NAME);
        return PTR_ERR(crypto_class);
    }

    crypto_device = device_create(crypto_class, NULL, MKDEV(major_number, 0), NULL, DEVICE_NAME);
    if (IS_ERR(crypto_device)) {
        class_destroy(crypto_class);
        unregister_chrdev(major_number, DEVICE_NAME);
        return PTR_ERR(crypto_device);
    }

    printk(KERN_INFO "Crypto: Device created successfully (major %d)\n", major_number);
    return 0;
}

static void __exit crypto_exit(void) {
    device_destroy(crypto_class, MKDEV(major_number, 0));
    class_unregister(crypto_class);
    class_destroy(crypto_class);
    unregister_chrdev(major_number, DEVICE_NAME);
    printk(KERN_INFO "Crypto: Module unloaded\n");
}

module_init(crypto_init);
module_exit(crypto_exit);
