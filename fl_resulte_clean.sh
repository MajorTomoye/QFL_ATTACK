#!/bin/bash

# 删除指定目录下的所有文件
rm -rf plots/cifar10/attack_fedlearn/*
rm -rf plots/tiny-imagenet/attack_fedlearn/*
rm -rf results/cifar10/attack_fedlearn/*
rm -rf results/tiny-imagenet/attack_fedlearn/*
rm -rf logs/cifar10/attack_fedlearn/*
rm -rf logs/tiny-imagenet/attack_fedlearn/*
rm -rf models/cifar10/attack_fedlearn/*
rm -rf models/tiny-imagenet/attack_fedlearn/*

# 输出操作完成信息
echo "All specified directories have been cleaned."
